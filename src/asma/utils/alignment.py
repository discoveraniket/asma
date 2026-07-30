import re
from typing import List

def find_source_sentences(document_text: str, value: str, reasoning: str, limit: int = 999) -> List[str]:
    """
    Analyzes document text, segments it into sentences, and identifies the sentences
    that most closely align with the extracted value and reasoning based on a tiered
    matching strategy.
    
    Tiers:
    1. Direct Substring Match (Ctrl+F style)
    2. Fuzzy Keyword Match (Token overlap)
    3. Reasoning Match (LLM thought overlap)
    
    Args:
        document_text: The full clean text of the document.
        value: The extracted target variable value.
        reasoning: The LLM's reasoning text.
        limit: Maximum number of source sentences to return.
        
    Returns:
        List of matching sentences.
    """
    if not document_text:
        return []
        
    # Segment document_text into individual sentences using re
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document_text)
    
    val_lower = (value or "").lower().strip()
    reason_lower = (reasoning or "").lower().strip()
    
    # -------------------------------------------------------------
    # Tier 1: Direct Substring Match (Ctrl+F)
    # -------------------------------------------------------------
    exact_matches = []
    if val_lower and len(val_lower) >= 2:
        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 8:
                continue
            if val_lower in s_clean.lower():
                exact_matches.append(s_clean)
                
    if exact_matches:
        return exact_matches[:limit]
        
    # Helpers for cleaning and word sets used in Tier 2 and Tier 3
    def clean_string(s: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\-\.\,\±]', ' ', (s or "").lower())).strip()

    def get_words(s: str) -> set:
        cleaned = clean_string(s)
        return {w for w in cleaned.split(' ') if len(w) > 2}
        
    # -------------------------------------------------------------
    # Tier 2: Fuzzy Keyword Match (Fallback A)
    # -------------------------------------------------------------
    fuzzy_matches = []
    words_v = get_words(val_lower) if val_lower else set()
    
    if words_v:
        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 8:
                continue
            s_lower = s_clean.lower()
            words_s = get_words(s_lower)
            intersect = words_s.intersection(words_v)
            if intersect:
                # Score based on word overlap ratio
                score = (len(intersect) / len(words_v)) * 100
                if score > 20:
                    fuzzy_matches.append((s_clean, score))
                    
    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in fuzzy_matches[:limit]]
        
    # -------------------------------------------------------------
    # Tier 3: Reasoning-only Fallback (Fallback B)
    # -------------------------------------------------------------
    reasoning_matches = []
    words_r = get_words(reason_lower) if reason_lower else set()
    
    if words_r:
        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 8:
                continue
            s_lower = s_clean.lower()
            if reason_lower in s_lower or s_lower in reason_lower:
                reasoning_matches.append((s_clean, 200))
            else:
                words_s = get_words(s_lower)
                intersect = words_s.intersection(words_r)
                if intersect:
                    score = (len(intersect) / len(words_r)) * 100
                    if score > 20:
                        reasoning_matches.append((s_clean, score))
                        
    if reasoning_matches:
        reasoning_matches.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in reasoning_matches[:limit]]
        
    return []


def generate_anchor_ids(field_name: str, count: int) -> List[str]:
    """
    Generates URL/DOM-safe sentence anchor identifiers for a given field name and sentence count.
    
    Example:
        generate_anchor_ids("Phage Name", 2) -> ["phage-name-anchor-1", "phage-name-anchor-2"]
    """
    if not field_name:
        field_safe = "anchor"
    else:
        field_safe = "".join([c if c.isalnum() else "-" for c in field_name.lower()]).strip("-")
        while "--" in field_safe:
            field_safe = field_safe.replace("--", "-")
        if not field_safe:
            field_safe = "anchor"

    return [f"{field_safe}-anchor-{i+1}" for i in range(count)]

