import re
import os
import sqlite3
import time
import json
import ast
import sys
import logging
import pandas as pd
import lmstudio as lms
import textwrap
import io
import matplotlib.pyplot as plt
from PIL import Image
from asma.core.parser import parse_bioc_to_llm_markdown

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extraction_pipeline")

# Configure standard output to handle UTF-8/unicode characters gracefully on Windows
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- ENVIRONMENT DETECTOR & DISPLAY FALLBACKS ---
try:
    from IPython.display import display, Markdown, clear_output
    IS_IPYTHON = True
except ImportError:
    IS_IPYTHON = False
    def display(x):
        if isinstance(x, pd.DataFrame):
            print(x.to_string())
        else:
            print(str(x))
    def Markdown(x):
        print(str(x))
    def clear_output(wait=False):
        pass

# --- CONFIGURATION CONSTANTS ---
JSON_PATH = "d:/Codes/GitHub/asma-workspace/litsift/server/cache/pmc_fetches/401d6d3929e951aaf358dc49e42a73c957ac495eeea976cc71647b19543f3b38.json"
MODEL_NAME = "google/gemma-4-e2b-qat"
BASE_URL = "localhost:1234"
GT_PATH = "d:/Codes/GitHub/asma-workspace/AM-AS/extracted_results_multipass.csv"
OUTPUT_CSV_PATH = "d:/Codes/GitHub/asma-workspace/AM-AS/extracted_results_multipass.csv"
MAX_ITERATIONS = 1  # Number of cells to extract in this run


# --- PROMPT TEMPLATES (Grouped Together) ---

SYSTEM_PROMPT_TEMPLATE = (
    "**SYSTEM INSTRUCTIONS**\n\n"
    "You are an expert data extraction, database schema and normalization assistant specialized in systematic reviews.\n"
    "Your core task is to extract highly accurate, to-the-point variable values from the provided research papers.\n\n"
    "RULES:\n"
    "* Always execute internal reasoning step-by-step before arriving at a conclusion.\n"
    "* Rely strictly on the facts explicitly reported in the document. Do not guess, or hallucinate.\n\n"
    "**Here is the research document:**\n\n"
    "<document>\n{document}\n</document>"
)

WARMUP_MESSAGE = (
    "**HANDSHAKE PROTOCOL**: \n\nAcknowledge that you have processed the system prompt "
    "and the embedded <document>. Do not perform any data extraction, analysis, "
    "or step-by-step reasoning yet. Respond with exactly the JSON array: ['Ready']"
)

P1_FACT_EXTRACTION_TEMPLATE = (
    "### Task\n"
    "{conditions}\n\n"
    "Find the relevant value(s) for '{target_column}'.\n\n"
    "Example style reference: '{example_format}'. (do NOT use this value, only match its format/style)\n\n"
    
    "### Researcher Reading Protocol\n"
    "Simulate a human researcher by scanning the document sections in this specific priority order:\n"
    "1. Abstract / Executive Summary / Introduction: Establish high-level context and core definitions.\n"
    "2. Main Findings, Data, & Exhibits: Analyze core data summaries, tables, and primary results.\n"
    "3. Methodology, Process, & Setup: Check here for technical frameworks, procedures, or underlying data sources.\n"
    "4. Discussion, Conclusion, & Appendices: Use as a fallback for secondary mentions or supplementary details.\n\n"
    
    "### Core Extraction Precision Rules\n"
    "- **Target Core Findings:** Extract ONLY the specific value(s) explicitly confirmed and studied for the target entity described in the criteria.\n"
    "- **Exclude Comprehensive Background:** Do NOT dump broad screening lists, negative controls, general literature baselines, or unconfirmed testing sets unless they are the specific focus of the query.\n"
    "- **Consolidate Specificity:** Focus on the primary, definitive representations of the data rather than enumerating every minor repetitive variant or raw table entry.\n\n"
    "- **Deduplicate & Normalize:** Always prefer the full, unabbreviated canonical name. Do not include abbreviated duplicates of the same item.\n\n"

    "### Thinking Guidelines\n"
    "Process your internal reasoning using the following sequential pattern:\n"
    "1. Cleaned Question: Rephrase the target query for absolute clarity.\n"
    "2. Source Quotes: Copy the verbatim text segments containing the target facts.\n"
    "3. Synthesis: Align the raw facts with the requested format.\n\n"
    
    "### OUTPUT FORMAT EXAMPLES\n"
    "- IF VALUES ARE FOUND:\n"
    "{{\n"
    '  "extracted_values": ["value A", "value B", ... etc]\n'
    "}}\n\n"
    "- IF NO VALUES ARE FOUND:\n"
    "{{\n"
    '  "extracted_values": []\n'
    "}}\n\n"
    
    "### FINAL INSTRUCTIONS\n"
    "Provide ONLY the valid JSON object using the exact key 'extracted_values'."
)

P2_DECISION_TREE_TEMPLATE = (
    "### ROLE & SCOPE\n\n"
    "You are a Database Normalization Clustering Engine.\n"
    "Your ONLY function is to group a provided list of candidate strings into row arrays for database storage.\n"
    "DO NOT extract new text from the paper.\n"
    "DO NOT summarize findings.\n"
    "DO NOT put extracted attribute data into the JSON arrays.\n\n"

    "### DATABASE OBJECTIVE\n"
    "Target Column: '{target_column}'\n\n"
    "Candidates to group: {candidates_str}\n\n"
    "Subsequent columns evaluated for collisions: {blank_fields_str}.\n\n"
    "Goal: Minimize database rows by combining candidates that share identical or empty data profiles into a single row array.\n\n"
    
    "### ROW ALLOCATION RULES\n\n"
    "1. GROUP together: Candidates with identical values or NO specific data in subsequent columns.\n\n"
    "2. SEPARATE: Candidates with unique, conflicting values that would cause data corruption if combined.\n\n"
    
    "### OUTPUT CONSTRAINTS & FORMAT\n"
    "The arrays inside the JSON object MUST contain ONLY exact string matches from the 'Candidates to group' list.\n\n"

    "### EXAMPLES OF OUTPUT FORMAT\n\n"
    "If the candidates were {candidates_mock_str}, your final JSON must look exactly like one of these structures:\n\n"
    "- IF ALL CAN SHARE A SINGLE ROW (Maximum Efficiency):\n"
    "{ex_group_all}\n"
    "- IF ALL REQUIRE DISTINCT ROWS (Conflicting Data):\n"
    "{ex_split_all}\n"
    "- IF SOME SHARE A ROW AND OTHERS DIFFER:\n"
    "{ex_mixed}\n\n"
    
    "### ANALYSIS & EXECUTION\n\n"
    "1. Perform a brief text comparison of candidate attributes across the subsequent columns.\n\n"
    "2. Conclude your response with the final JSON object mapping row keys strictly to candidate string arrays.\n"
)


# --- HELPER FUNCTIONS ---

def get_paper_content_from_json(json_path):
    """Loads and parses PMC Bioc JSON to raw Markdown context."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing PMC JSON payload at: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            bioc_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}") from e
    return parse_bioc_to_llm_markdown(bioc_data)


def find_first_empty_cell(df):
    """Finds the row and column coordinates of the first empty cell in the dataframe."""
    for idx, row in df.iterrows():
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val == "" or val is None:
                return idx, col
    return None, None


def find_example_value(df, target_column):
    """Retrieves an existing filled example value for the column to serve as a style guide."""
    filled_values = df[
        df[target_column].notna() &
        (df[target_column] != "") &
        (df[target_column] != "Not specified")
    ][target_column]
    if not filled_values.empty:
        return filled_values.iloc[0]
    return None


def build_conditions_and_example(df):
    """Extracts context conditions from currently populated columns in the active row."""
    row_idx, target_column = find_first_empty_cell(df)
    if row_idx is None or target_column is None:
        return None, None, None, None
    row = df.loc[row_idx]
    filled_conditions = []
    for col in df.columns:
        if col != target_column:
            val = row[col]
            if not (pd.isna(val) or val == "" or val is None):
                filled_conditions.append(f"the '{col}' is '{val}'")
    example_val = find_example_value(df, target_column)
    return row_idx, target_column, filled_conditions, example_val


def format_extraction_prompt(target_column, conditions, example_val, template_str):
    """Fills templates for Pass 1 extraction prompts."""
    conditions_str = ""
    if conditions:
        conditions_str = "When " + "\nand \n".join(conditions)
    example_str = ""
    if example_val:
        example_str = example_val
    return template_str.format(
        target_column=target_column,
        conditions=conditions_str,
        example_format=example_str
    ).strip()


def split_llm_response(text):
    """Extracts internal thoughts and the final output block from model response."""
    if not text:
        return "", ""
    end_pattern = r"__LM_STUDIO_INTERNAL_LSEP_SYNTHETIC_REASONING_END_[a-f0-9]+__"
    matches = list(re.finditer(end_pattern, text))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        thought = text[:start]
        content = text[end:]
        return thought.strip(), content.strip()
    return "", text.strip()


def extract_values_from_decision(text):
    """Parses candidate output text prefix structures (V1:, V2: etc.)."""
    if not text:
        return ""
    lines = text.split('\n')
    values_start = False
    values_list = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if "values:" in cleaned.lower():
            values_start = True
            continue
        if values_start:
            val = cleaned
            val = re.sub(r'^V\d+:\s*', '', val, flags=re.IGNORECASE)
            if val.startswith(("-", "*", "•", "1.", "2.", "3.", "4.")):
                val = val.split(None, 1)[-1]
            values_list.append(val.strip().strip('"').strip("'"))
    if not values_list:
        for line in lines:
            cleaned = line.strip()
            if cleaned and not cleaned.lower().startswith("analysis:"):
                val = cleaned
                val = re.sub(r'^V\d+:\s*', '', val, flags=re.IGNORECASE)
                if val.startswith(("-", "*", "•")):
                    val = val.split(None, 1)[-1]
                values_list.append(val.strip().strip('"').strip("'"))
    return "\n".join(values_list)


def parse_programmatically(text):
    """Parses raw text to json dictionary/list cleanly."""
    if not text:
        raise ValueError("Cannot parse empty response text.")
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No valid JSON object boundaries found.")
        
    raw_obj = text[start_idx:end_idx + 1]
    
    try:
        parsed = json.loads(raw_obj)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
        
    try:
        parsed = ast.literal_eval(raw_obj)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
        
    raise ValueError("Failed to parse string segment as a JSON object.")


def parse_with_llm(text, model, split_func):
    """Uses LLM as a fallback converter if programmatically parsing JSON fails."""
    prompt = (
        "Convert this text into a clean valid JSON object. "
        "Output only the raw JSON object. No markdown, no conversation.\n"
        f"Text: {text}"
    )
    response = model.respond(prompt)
    _, clean_text = split_func(response.content)
    return parse_programmatically(clean_text)


def extract_json(text, llm_model, split_func):
    """Extracts JSON with local fallback engine."""
    try:
        return parse_programmatically(text)
    except ValueError:
        return parse_with_llm(text, llm_model, split_func)


def insert_and_split(df, row_idx, col_name, extracted_values):
    """Duplicates row properties and inserts normalized candidates across split rows."""
    if not extracted_values:
        if df[col_name].dtype != "object":
            df[col_name] = df[col_name].astype(object)
        df.at[row_idx, col_name] = "Not specified"
        return df
    original_row = df.iloc[row_idx].copy()
    new_rows = []
    for val in extracted_values:
        row_copy = original_row.copy()
        row_copy[col_name] = val
        new_rows.append(row_copy)
    new_rows_df = pd.DataFrame(new_rows)
    df_before = df.iloc[:row_idx]
    df_after = df.iloc[row_idx + 1:]
    return pd.concat([df_before, new_rows_df, df_after], ignore_index=True)


def df_to_image(df):
    """Saves structured data frame to a visual PIL Image."""
    temp_df = df.fillna('-').replace('', '-')
    max_lens = []
    for col in temp_df.columns:
        max_len = max(temp_df[col].astype(str).map(len).max(), len(str(col)))
        max_lens.append(max_len)
    total_len = sum(max_lens)
    col_widths = [max(0.03, w / total_len) for w in max_lens]
    total_normalized = sum(col_widths)
    col_widths = [w / total_normalized for w in col_widths]
    
    wrapped_headers = []
    for col, width_ratio in zip(temp_df.columns, col_widths):
        wrap_width = max(12, int(width_ratio * 150))
        wrapped_headers.append(textwrap.fill(str(col), width=wrap_width))
        
    wrapped_data = []
    for row in temp_df.values:
        wrapped_row = []
        for val, width_ratio in zip(row, col_widths):
            wrap_width = max(12, int(width_ratio * 150))
            wrapped_row.append(textwrap.fill(str(val), width=wrap_width))
        wrapped_data.append(wrapped_row)
        
    row_height = 1.8
    fig_width = 30
    fig_height = len(df) * row_height + 4.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    tbl = ax.table(
        cellText=wrapped_data,
        colLabels=wrapped_headers,
        colWidths=col_widths,
        loc='center',
        cellLoc='left'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 1.8)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor('#dcdcdc')
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2b3e50')
        else:
            if row_idx % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def build_decision_tree_prompt(target_column, candidates, blank_fields):
    """Constructs the clustering decision prompt for Pass 2."""
    # Unwrap candidate structures defensively
    if isinstance(candidates, dict):
        candidates = candidates.get("extracted_values", [])
    elif isinstance(candidates, str):
        try:
            parsed = json.loads(candidates)
            candidates = parsed.get("extracted_values", []) if isinstance(parsed, dict) else parsed
        except Exception:
            candidates = []

    if not isinstance(candidates, list) or not candidates:
        return None

    candidates_str = ", ".join([f"'{c}'" for c in candidates])
    blank_fields_str = ", ".join([f"'{f}'" for f in blank_fields])
    
    # Ensure mock examples always have at least 3 items to prevent IndexErrors
    N_mock = max(len(candidates), 3)
    c_mock = [f"Item {i}" for i in range(1, N_mock + 1)]
    
    ex_group_all = json.dumps({"Row 1": c_mock}, indent=2)
    split_dict = {f"Row {i+1}": [c] for i, c in enumerate(c_mock)}
    ex_split_all = json.dumps(split_dict, indent=2)
    
    mixed_dict = {"Row 1": [c_mock[0], c_mock[1]]}
    for i, c in enumerate(c_mock[2:], 2):
        mixed_dict[f"Row {i}"] = [c]
    ex_mixed = json.dumps(mixed_dict, indent=2)

    prompt = P2_DECISION_TREE_TEMPLATE.format(
        target_column=target_column,
        candidates_str=candidates_str,
        blank_fields_str=blank_fields_str,
        candidates_mock_str=", ".join(c_mock),
        ex_group_all=ex_group_all,
        ex_split_all=ex_split_all,
        ex_mixed=ex_mixed
    )
    return prompt


# --- MAIN PIPELINE WORKFLOW ---

def run_extraction_pipeline():
    try:
        logger.info(f"Loading paper text from JSON: {JSON_PATH}...")
        paper_text = get_paper_content_from_json(JSON_PATH)
        logger.info(f"Loaded and parsed paper text: {len(paper_text)} characters.")
    except Exception as e:
        logger.critical(f"Failed to read/parse document: {e}")
        sys.exit(1)

    try:
        logger.info(f"Loading ground truth data from: {GT_PATH}...")
        tbl_gt = pd.read_csv(GT_PATH)
        tbl_extract = tbl_gt.astype(object)
    except Exception as e:
        logger.critical(f"Failed to load target CSV: {e}")
        sys.exit(1)
    
    try:
        logger.info("Connecting to LM Studio Client...")
        client = lms.Client(BASE_URL)
        model = client.llm.model(MODEL_NAME)
        chat = lms.Chat()
    except Exception as e:
        logger.critical(f"Connection to LM Studio failed: {e}. Check if LM Studio is running on {BASE_URL}.")
        sys.exit(1)

    # Pre-warm context construction
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(document=paper_text)

    logger.info("Pre-warming model context...")
    chat._messages[:] = chat._messages[:0]
    chat.add_system_prompt(system_prompt)
    chat.add_user_message(WARMUP_MESSAGE)
    
    try:
        warmup_reply = model.respond(chat)
        logger.info(f"Handshake Reply: {warmup_reply.content.strip()}")
    except Exception as e:
        logger.critical(f"Inference error during handshake: {e}")
        sys.exit(1)

    # Output Token size for diagnostics
    try:
        formatted = model.apply_prompt_template(chat)
        tokens = model.tokenize(formatted)
        logger.info(f"Pre-warm token size: {len(tokens)}")
    except Exception as e:
        logger.warning(f"Could not calculate token sizes: {e}")

    gif_frames = []

    logger.info(f"Starting extraction loops (Max Iterations: {MAX_ITERATIONS})...")
    for i in range(MAX_ITERATIONS):
        row, column, conditions, example_val = build_conditions_and_example(tbl_extract)
        if row is None or column is None:
            logger.info("Extraction completed! No remaining blank cells found.")
            break

        logger.info(f"--- [Iteration {i+1}] Extracting '{column}' for Row {row} ---")
        
        # --- PASS 1: Fact Extraction ---
        logger.info("Running Fact Extraction...")
        p1_prompt = format_extraction_prompt(column, conditions, example_val, P1_FACT_EXTRACTION_TEMPLATE)
        
        chat._messages[:] = chat._messages[:1]
        chat.add_user_message(p1_prompt)
        
        try:
            p1_response = model.respond(chat)
            p1_raw = p1_response.content
        except Exception as e:
            logger.error(f"Inference failed in Pass 1: {e}")
            continue

        p1_thought, p1_text = split_llm_response(p1_raw)
        if not p1_text:
            p1_text = p1_raw
        
        logger.info(f"Pass 1 Raw Output:\n{p1_text}")
        
        p1_text_clean = p1_text.replace("\xa0", " ")
        try:
            p1_data = extract_json(p1_text_clean, model, split_llm_response)
            p1_values = p1_data.get("extracted_values", [])
        except Exception as e:
            logger.error(f"Pass 1 Parsing Error: {e}. Defaulting to empty candidates.")
            p1_values = []
            
        has_multiple_candidates = len(p1_values) > 1
        
        # --- PASS 2: Relational Split Decision ---
        if has_multiple_candidates:
            logger.info(f"Multiple candidates detected ({p1_values}). Running Pass 2...")
            current_row = tbl_extract.iloc[row]
            blank_cols = [
                col for col in tbl_extract.columns 
                if col != column and (pd.isna(current_row[col]) or current_row[col] == "" or current_row[col] is None)
            ]

            p2_prompt = build_decision_tree_prompt(
                target_column=column,
                candidates=p1_values,
                blank_fields=blank_cols
            )

            chat._messages[:] = chat._messages[:1]
            chat.add_user_message(p2_prompt)
            
            try:
                p2_response = model.respond(chat)
                p2_raw = p2_response.content
            except Exception as e:
                logger.error(f"Inference failed in Pass 2: {e}")
                data = p1_values
                continue

            p2_thought, p2_text = split_llm_response(p2_raw)
            if not p2_text:
                p2_text = p2_raw
                
            logger.info(f"Pass 2 Raw Output:\n{p2_text}")
            
            p2_text_clean = p2_text.replace("\xa0", " ")
            try:
                p2_data = extract_json(p2_text_clean, model, split_llm_response)
                data = [", ".join(v) if isinstance(v, list) else str(v) for v in p2_data.values()]
            except Exception as e:
                logger.error(f"Pass 2 Parsing Error: {e}. Falling back to default split.")
                data = p1_values
        else:
            logger.info("Single candidate or no values found. Skipping Pass 2 clustering.")
            data = p1_values

        # Insert extracted values and display table state
        tbl_extract = insert_and_split(tbl_extract, row, column, data)
        display(tbl_extract)
        
        try:
            frame = df_to_image(tbl_extract)
            gif_frames.append(frame)
        except Exception as e:
            logger.warning(f"Visual rendering skipped: {e}")

    # Save output to CSV
    try:
        logger.info(f"Saving finalized extraction results to {OUTPUT_CSV_PATH}...")
        tbl_extract.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info("Pipeline run successfully completed.")
    except Exception as e:
        logger.error(f"Failed to save CSV output: {e}")


if __name__ == "__main__":
    run_extraction_pipeline()
