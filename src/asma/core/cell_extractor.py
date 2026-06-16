import re
import json
import ast
from typing import Optional, List, Tuple

DEFAULT_CELL_TEMPLATE = """### Your Task:
{conditions}

**QUESTION**: Find the best value/values for '{target_column}'.
{example_format}

### Thought Channel Guidelines (Output in <|channel>thought):
1. Cleaned Question: Rephrase the user's question in the context of this specific document.
2. Source Quotes: Copy the exact sentences/paragraphs from the document that contain the answer.
3. Synthesis: Extract the final Answer Value from your quotes, matching the style/format of the example if provided, but extracting the actual value from the source quotes"""


def build_extraction_prompt(
    target_column: str,
    conditions: List[str],
    example_val: Optional[str] = None,
    template_str: Optional[str] = None
) -> str:
    """
    Formulates the prompt specifying what target variable to extract, taking current row variables
    as active context conditions.
    """
    if not template_str:
        template_str = DEFAULT_CELL_TEMPLATE
    
    conditions_str = ""
    if conditions:
        conditions_str = "When " + "\nand \n".join(conditions)
        
    example_str = ""
    if example_val:
        example_str = f"- Example style reference (do NOT use this value, only match its format/style): '{example_val}'"
        
    return template_str.format(
        target_column=target_column,
        conditions=conditions_str,
        example_format=example_str
    ).strip()


def split_llm_response(text: str) -> Tuple[str, str]:
    """
    Robustly separates the thought channel block (<|channel>thought ... <channel|>) from the final output response content.
    """
    end_pattern = r"(?:<channel\|>)|(?:<\/channel>)|(?:channel\|>)"
    matches = list(re.finditer(end_pattern, text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        thought = text[:start]
        content = text[end:]
        
        start_pattern = r"(?:<\|)?channel>thought"
        thought = re.sub(start_pattern, "", thought, flags=re.IGNORECASE)
        return thought.strip(), content.strip()
        
    return "", text.strip()


def extract_json_array(text: str) -> List[str]:
    """
    Strips markdown formatting backticks and securely parses string segments representing list structures
    into standard Python string arrays.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No valid JSON array boundaries found.")
        
    raw_array = text[start_idx:end_idx + 1]
    
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
