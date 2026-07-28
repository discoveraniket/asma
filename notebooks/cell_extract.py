# %%
import re
import os
import sqlite3
import time
import json
import ast
import pandas as pd
import lmstudio as lms
from IPython.display import display, Markdown, clear_output
import textwrap
import io
import matplotlib.pyplot as plt
from PIL import Image

DB_PATH = "d:/Codes/GitHub/asma-workspace/litsift/server/papers.db"
MODEL_NAME = "google/gemma-4-e2b-qat"
BASE_URL = "localhost:1234"
GT_PATH = "d:/Codes/GitHub/asma-workspace/AM-AS/gt2.csv"
OUTPUT_CSV_PATH = "d:/Codes/GitHub/asma-workspace/AM-AS/extracted_results.csv"
PAPER_ID = "Paper_38956568"

# %%
def get_paper_content(db_path, paper_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT raw_content FROM papers WHERE id = ?", (paper_id,))
        row = cursor.fetchone()
        if row:
            return str(row[0])
    raise ValueError(f"Paper {paper_id} not found.")

client = lms.Client(BASE_URL)
model = client.llm.model(MODEL_NAME)

# %%
paper_text = get_paper_content(DB_PATH, PAPER_ID)
chat = lms.Chat()
document_context = (
    "You are a document reading assistant. Here is the research article:\n\n"
    f"<document>\n{paper_text}\n</document>"
)
chat.add_system_prompt(document_context)
chat.add_user_message('Confirm you have loaded the document by responding with exactly the JSON array: ["Ready"]')
pre_warm_reply = model.respond(chat)
chat._messages[:] = chat._messages[:2]
chat.add_assistant_response(pre_warm_reply.content)
Markdown(pre_warm_reply.content)

# %%
tbl_gt = pd.read_csv(GT_PATH)
tbl_extract = tbl_gt.iloc[[0]].copy()
blank_row = pd.DataFrame([[""] * len(tbl_extract.columns)], columns=tbl_extract.columns)
tbl_extract = pd.concat([tbl_extract, blank_row], ignore_index=True)
tbl_extract

# %%
def find_first_empty_cell(df):
    for idx, row in df.iterrows():
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val == "" or val is None:
                return idx, col
    return None, None

def find_example_value(df, target_column):
    filled_values = df[
        df[target_column].notna() &
        (df[target_column] != "") &
        (df[target_column] != "Not specified")
    ][target_column]
    if not filled_values.empty:
        return filled_values.iloc[0]
    return None

def build_extraction_prompt(df):
    if len(df) == 0:
        df.loc[0] = [""] * len(df.columns)
    row_idx, target_column = find_first_empty_cell(df)
    if row_idx is None or target_column is None:
        return None, None, "The table is already fully populated!"
    row = df.loc[row_idx]
    filled_conditions = []
    for col in df.columns:
        if col != target_column:
            val = row[col]
            if not (pd.isna(val) or val == "" or val is None):
                filled_conditions.append(f"the '{col}' is '{val}'")
    prompt_lines = ["### Your Task:\n"]
    if filled_conditions:
        prompt_lines.append("When " + "\nand \n".join(filled_conditions))
        prompt_lines.append(f"\n**QUESTION**: Find the best value/values for '{target_column}'.")
    else:
        prompt_lines.append(f"\n**QUESTION**: Find the best value/values for '{target_column}'.")
    example_val = find_example_value(df, target_column)
    if example_val:
        prompt_lines.append(f"- Example format reference: '{example_val}'")
    # prompt_lines.extend([
    #     "\n**Instructions for your thinking channel (`thought`)**:",
    #     f"* First, rephrase the question clearly. Make clear what '{target_column}' means in this context and what specific data types/formats are expected.",
    #     "* Reiterate the cleaned question in your thinking. Do not output it in your final result.",
    #     "* Identify and scan the candidate sections to synthesize the answer.",
    #     "* Return multiple values only if they are explicitly supported by the text."
    # ])

    # prompt_lines.extend([
    #     "\n**Thinking Guidelines**:",
    #     f"1. Rephrase what '{target_column}' means in this context and note what data type/format is expected.",
    #     "2. Scan the document sections and locate candidates for the value.",
    #     "3. Carefully evaluate if there is truly more than one distinct value supported by the text before outputting multiple elements."
    # ])

    prompt_lines.extend([
        "\n### Thought Channel Guidelines (Output in <|channel>thought):",
        "1. Cleaned Question: Rephrase the user's question in the context of this specific document.",
        "2. Source Quotes: Copy the exact sentences/paragraphs from the document that contain the answer.",
        "3. Synthesis: Extract the final Answer Value from your quotes, matching the example format if provided."
    ])

    return row_idx, target_column, "\n".join(prompt_lines)

def split_llm_response(text):
    # Matches common variations of the closing thought tag
    end_pattern = r"(?:<channel\|>)|(?:<\/channel>)|(?:channel\|>)"
    
    matches = list(re.finditer(end_pattern, text, re.IGNORECASE))
    if matches:
        # Get the span of the very last closing tag
        last_match = matches[-1]
        start, end = last_match.span()
        
        # Everything before the last closing tag is part of the thinking process
        thought = text[:start]
        # Everything after the last closing tag is the final output
        content = text[end:]
        
        # Clean up any start tags inside the thought text
        start_pattern = r"(?:<\|)?channel>thought"
        thought = re.sub(start_pattern, "", thought, flags=re.IGNORECASE)
        
        return thought.strip(), content.strip()
        
    return "", text.strip()

def extract_json_array(text):
    text = text.strip()
    
    # Strip markdown block formatting if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    
    # If no square brackets exist, it is not a valid JSON array
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No valid JSON array boundaries found.")
        
    raw_array = text[start_idx:end_idx + 1]
    
    # Try parsing strictly
    try:
        parsed = json.loads(raw_array)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
        
    try:
        parsed = ast.literal_eval(raw_array)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
        
    raise ValueError("Failed to parse string segment as a JSON list.")

def insert_and_split(df, row_idx, col_name, extracted_values):
    if not extracted_values:
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
        wrapped_data.append(wrapped_row) # Fixed typo here
        
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


# %%
system_prompt = (
    "**SYSTEM INSTRUCTIONS**\n\n"
    "You are an expert data extraction assistant.\n"
    "Your task is to extract key information from research articles to help create a structured dataset.\n\n"
    "RULES:\n"
    "* Always think step by step in your thinking channel (`thought`) before responding.\n"
    "* Extract to-the-point data from the provided document.\n"
    "* If a specific field is not mentioned, output [\"Not specified\"]. Do not guess or hallucinate.\n"
    "* Output your response *strictly* as a valid, parseable JSON array of strings.\n\n"
    "DECISION RULE: SINGLE VS. MULTIPLE VALUES\n\n"
    "Choose how many elements to return in the JSON array based on this rule:\n"
    "1. Use MULTIPLE elements (e.g., [\"entity-A\", \"entity-B\"]) ONLY when extracting distinct, independent entities that require separate rows in the database.\n"
    "2. Use a SINGLE element (e.g., [\"head: 60nm, tail: 20nm\" or \"37°C and 42°C\"]) when extracting multiple attributes, measurements, ranges, or descriptors of a SINGLE entity."
)

chat._messages[:] = chat._messages[:3]
chat.add_system_prompt(system_prompt)
Markdown(system_prompt)

# %%
row, column, user_prompt = build_extraction_prompt(tbl_extract)
Markdown(user_prompt)

# %%
MAX_ITERATIONS = 20
gif_frames = []

for i in range(MAX_ITERATIONS):
    row, column, user_prompt = build_extraction_prompt(tbl_extract)
    if user_prompt == "The table is already fully populated!":
        print("Extraction completed!")
        break
    chat._messages[:] = chat._messages[:4]
    chat.add_user_message(user_prompt)

    response = model.respond(chat)
    thinking, content = split_llm_response(response.content)

    try:
        data = extract_json_array(content)
    except Exception as e:
        print(f"Malformed JSON detected for '{column}'. Requesting self-correction...")
        chat.add_assistant_response(response.content)
        
        max_retries = 2
        success = False
        
        for attempt in range(max_retries):
            chat.add_user_message(
                "Your previous response was not a valid JSON array of strings. "
                "Please output ONLY the JSON array (e.g. [\"value1\"]). "
                "Do not include conversational text or markdown formatting."
            )

            repair_response = model.respond(chat)
            _, repair_content = split_llm_response(repair_response.content)
            
            try:
                data = extract_json_array(repair_content)
                success = True
                break
            except Exception:
                chat.add_assistant_response(repair_response.content)
        
        if not success:
            print("Self-correction failed after maximum retries. Falling back to raw output.")
            data = [content]

    tbl_extract = insert_and_split(tbl_extract, row, column, data)

    try:
        frame = df_to_image(tbl_extract)
        gif_frames.append(frame)
    except Exception:
        pass
    
    # Clear first, then print everything you want to keep visible
    clear_output(wait=True)
    print(f"Extracted {column}: {data}")
    display(tbl_extract)
    print(thinking)


# %%
from IPython.display import Image as IPImage, display
if gif_frames:
    gif_frames[0].save(
        "table_extraction_progress.gif",
        save_all=True,
        append_images=gif_frames[1:],
        optimize=False,
        duration=800,
        loop=0
    )

# Display the GIF
display(IPImage(filename="table_extraction_progress.gif"))

# %%
# tbl_extract.to_csv(OUTPUT_CSV_PATH, index=False)


