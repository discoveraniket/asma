import re
from typing import List

def find_source_sentences(document_text: str, value: str, reasoning: str, limit: int = 2) -> List[str]:
    """
    Analyzes document text, segments it into sentences, and identifies the sentences
    that most closely align with the extracted value and reasoning based on overlap scores.
    
    Args:
        document_text: The full clean text of the document.
        value: The extracted target variable value.
        reasoning: The LLM's reasoning text.
        limit: Maximum number of source sentences to return.
        
    Returns:
        List of matching sentences sorted by overlap score descending.
    """
    if not document_text:
        return []
        
    # Segment document_text into individual sentences using re
    # Split by standard sentence delimiters (. ! ?) keeping abbreviations/decimal numbers in mind
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document_text)
    
    def clean_string(s: str) -> str:
        # Convert to lowercase and normalize punctuation/spaces
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\-\.\,\±]', ' ', (s or "").lower())).strip()

    def get_words(s: str) -> set:
        cleaned = clean_string(s)
        return {w for w in cleaned.split(' ') if len(w) > 2}

    scored_sentences = []
    
    for sentence in sentences:
        s_clean = sentence.strip()
        if len(s_clean) < 8:
            continue
            
        score = 0
        s_lower = s_clean.lower()
        val_lower = (value or "").lower().strip()
        reason_lower = (reasoning or "").lower().strip()
        
        # 1. Exact value match as substring
        if val_lower and len(val_lower) >= 4:
            if val_lower in s_lower:
                score += 150
            else:
                words_v = get_words(val_lower)
                if words_v:
                    words_s = get_words(s_lower)
                    intersect = words_s.intersection(words_v)
                    score += (len(intersect) / len(words_v)) * 50
                    
        # 2. Exact or partial overlap of reasoning
        if reason_lower and len(reason_lower) >= 6:
            if reason_lower in s_lower or s_lower in reason_lower:
                score += 200
            else:
                words_r = get_words(reason_lower)
                if words_r:
                    words_s = get_words(s_lower)
                    intersect = words_s.intersection(words_r)
                    score += (len(intersect) / len(words_r)) * 100
                    
        if score > 20:
            scored_sentences.append((s_clean, score))
            
    # Sort matches by score descending
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in scored_sentences[:limit]]
