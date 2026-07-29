import json
import logging
from typing import Optional, List, Dict
from asma.interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)

# Ephemeral cache to store the last structuring prompt/response per paper
structuring_history = {}


STRUCTURE_PROMPT_TEMPLATE = """You are an expert data formatting assistant.
You are given a flat list of extracted fields from a scientific article. These fields describe characterizations or properties of target entities (specifically phages and host bacteria). Some fields might contain concatenated values, multiple items, or descriptions for multiple different phages (e.g., "Phage 4141: 15 min; Phage MJW: 12 min").

Your task is to analyze these extractions and split them into a list of structured records, where each record represents exactly ONE distinct phage/bacteria characterization instance.

Target Schema Fields:
{fields_list}

Extracted Key-Value Data:
{extractions_text}

Rules:
1. Output a valid, parseable JSON array of objects.
2. Each object in the array must contain keys matching the target schema fields exactly.
3. If a field has a single global value (e.g., targeted bacteria or paper details) that applies to all instances, 
duplicate that value for all records.
4. Split multi-value fields: Check if fields (such as sample collections, morphology, isolation sources, or dimensions) 
contain details for multiple entities. You must split and assign only the specific, concise value belonging to that single instance. 
Under no circumstances should you repeat a combined multi-entity description across both rows.
5. Correlate Transitive Mappings: Align fields using logical chain associations (e.g. If Phage A mapped to 'clinical stool', 
and the text states 'clinical stool from IDBG Hospital' and 'sewage from Nadia', then Phage A must be assigned ONLY 
'IDBG Hospital' for its collection location, and Phage B must be assigned ONLY 'Nadia'). 
Clean up any narrative text (e.g., extract "IDBG Hospital, Kolkata" instead of "clinical stool samples obtained from IDBG Hospital, Kolkata").
6. If a field's value is missing or not applicable for a specific instance, set its value to "".
7. Do not include markdown code block formatting (like ```json) or any conversational text around the JSON array. Output ONLY the raw JSON array.
"""

def extract_json_array(text: str) -> List[Dict[str, str]]:
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
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("Could not find JSON array bounds in the response.")
    
    json_str = text[start_idx:end_idx + 1]
    data = json.loads(json_str)
    if not isinstance(data, list):
        raise ValueError("Decoded JSON is not a list of objects.")
    return data

def structure_extractions(
    extractions: Dict[str, str],
    fields: List[str],
    llm_provider: LLMProvider,
    paper_id: Optional[str] = None,
    history_collector: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Takes flat key-value extractions and structures them into a list of relational records
    by calling the LLM provider to format and parse the contents.
    """
    if not extractions:
        return []

    # Format the template inputs
    fields_list = "\n".join(f"* {f}" for f in fields)
    extractions_text = "\n".join(f"- {k}: {v}" for k, v in extractions.items())
    
    prompt = STRUCTURE_PROMPT_TEMPLATE.format(
        fields_list=fields_list,
        extractions_text=extractions_text
    )
    
    system_instruction = "You are a precise data structuring assistant. You output only valid raw JSON arrays."
    
    try:
        raw_response = llm_provider.respond(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            stream=False
        )
        logger.info(f"Received structuring response: {raw_response}")
        
        history_entries = [
            {"role": "Pass 2 System (Structuring)", "content": system_instruction},
            {"role": "Pass 2 User (Structuring)", "content": prompt},
            {"role": "Pass 2 Assistant (Structuring)", "content": raw_response}
        ]

        if history_collector is not None:
            history_collector.extend(history_entries)

        if paper_id:
            structuring_history[paper_id] = history_entries

        
        # If response contains a thinking block or thought tag, we split it
        # Import split_llm_response here to avoid circular imports
        from asma.utils.text import split_llm_response
        _, parsed_content = split_llm_response(raw_response)
        
        records = extract_json_array(parsed_content)
        
        # Post-process records to ensure all requested fields are present, normalized as strings
        validated_records = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            validated_rec = {}
            for field in fields:
                val = rec.get(field, "")
                validated_rec[field] = str(val) if val is not None else ""
            validated_records.append(validated_rec)
            
        return validated_records
        
    except Exception as e:
        logger.error(f"Failed to structure extractions with LLM: {e}")
        # Return a single fallback record containing the raw values for robustness
        fallback_rec = {field: extractions.get(field, "") for field in fields}
        return [fallback_rec]
