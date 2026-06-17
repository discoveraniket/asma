import re
from typing import Dict, List

def clean_text(text: str) -> str:
    """
    Remove bracketed citations like [1], [1, 2], [3-5] and normalize whitespace.
    """
    # Remove bracketed citations like [1], [1, 2], [3-5]
    cleaned = re.sub(r'\[\d+(?:[\s,\u2013-]+\d+)*\]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def clean_text_keep_citations(text: str) -> str:
    """
    Normalize whitespace in a string but preserve bracketed citations like [1].
    """
    return re.sub(r'\s+', ' ', text).strip()

def extract_metadata_llm(infons: Dict[str, str]) -> List[str]:
    """
    Extract structured metadata from BioC passage infons dictionary and return a list of formatted lines.
    """
    lines = []
    
    doi = infons.get("article-id_doi")
    if doi:
        lines.append(f"DOI: https://doi.org/{doi}")
        
    pmc = infons.get("article-id_pmc")
    pmid = infons.get("article-id_pmid")
    if pmc: lines.append(f"PMC: {pmc}")
    if pmid: lines.append(f"PMID: {pmid}")
    
    volume = infons.get("volume")
    issue = infons.get("issue")
    year = infons.get("year")
    journal_parts = []
    if volume: journal_parts.append(f"Volume {volume}")
    if issue: journal_parts.append(f"Issue {issue}")
    if year: journal_parts.append(f"({year})")
    if journal_parts:
        lines.append(f"Published: {', '.join(journal_parts)}")
        
    authors = []
    for i in range(50):
        val = infons.get(f"name_{i}")
        if not val:
            break
        parts = dict(item.split(":", 1) for item in val.split(";") if ":" in item)
        name = f"{parts.get('given-names', '')} {parts.get('surname', '')}".strip()
        if name and name not in authors:
            authors.append(name)
    if authors:
        lines.append(f"Authors: {', '.join(authors)}")

    kwd = infons.get("kwd")
    if kwd:
        lines.append(f"Keywords: {kwd}")
        
    return lines

from typing import Tuple, Optional

BUILTIN_REASONING_TAGS = [
    ("<|channel>thought", "<channel|>"),  # LM Studio gemma-4-e2b
    ("<think>", "</think>"),              # DeepSeek models
    ("<thought>", "</thought>"),          # Generic thought tags
]

def split_llm_response(
    message: str,
    start_tag: Optional[str] = None,
    end_tag: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Splits raw LLM output into a reasoning thought block and the final response block.
    
    Args:
        message: The raw response string from the LLM.
        start_tag: Optional custom start tag (e.g. '<think>').
        end_tag: Optional custom end tag (e.g. '</think>').
        
    Returns:
        A tuple of (thought_content, response_content).
        thought_content is None if no thought block is found.
    """
    text = message.strip()
    
    # 1. Custom tag pair override
    if start_tag and end_tag:
        idx_start = text.lower().find(start_tag.lower())
        if idx_start != -1:
            idx_end = text.lower().find(end_tag.lower(), idx_start + len(start_tag))
            if idx_end != -1:
                thought = text[idx_start + len(start_tag):idx_end].strip()
                content = (text[:idx_start] + " " + text[idx_end + len(end_tag):]).strip()
                return thought, content
        return None, text

    # 2. Robust regex-based channel thought extraction
    # Support end tags like <channel|>, </channel>, channel|> case-insensitively
    end_pattern = r"(?:<channel\|>)|(?:<\/channel>)|(?:channel\|>)"
    matches = list(re.finditer(end_pattern, text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        thought = text[:start]
        content = text[end:]
        
        # Strip the start tag: <|channel>thought or similar variation
        start_pattern = r"(?:<\|)?channel>thought"
        thought = re.sub(start_pattern, "", thought, flags=re.IGNORECASE).strip()
        # Clean up any leading '<|' or '<' if left over
        thought = re.sub(r'^<\|?', '', thought).strip()
        return thought, content.strip()

    # 3. Lookup standard built-in tag pairs case-insensitively
    for s_tag, e_tag in BUILTIN_REASONING_TAGS:
        s_lower, e_lower = s_tag.lower(), e_tag.lower()
        text_lower = text.lower()
        idx_start = text_lower.find(s_lower)
        if idx_start != -1:
            idx_end = text_lower.find(e_lower, idx_start + len(s_tag))
            if idx_end != -1:
                thought = text[idx_start + len(s_tag):idx_end].strip()
                content = (text[:idx_start] + " " + text[idx_end + len(e_tag):]).strip()
                return thought, content

    # 4. Default fallback: No thought block detected
    return None, text
