import re
from typing import Dict, List, Tuple, Optional

def clean_text(text: str) -> str:
    """
    Remove bracketed citations like [1], [1, 2], [3-5] and normalize whitespace.
    """
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


def split_llm_response(
    message: str,
    start_tag: Optional[str] = None,
    end_tag: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Splits raw LLM output into a reasoning thought block and the final response block
    using LM Studio's synthetic reasoning end pattern.
    """
    if not message:
        return None, ""

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

    # 2. Check for <think>...</think> or <|channel>thought...<channel|> tags
    tag_match = re.search(r'<(?:think>|\|channel>thought)[\s\r\n]*([\s\S]*?)[\s\r\n]*?(?:</think>|<channel\|>)', text, re.IGNORECASE)
    if tag_match:
        thought = tag_match.group(1).strip()
        content = re.sub(r'<(?:think>|\|channel>thought)[\s\S]*?(?:</think>|<channel\|>)', '', text, flags=re.IGNORECASE).strip()
        return thought if thought else None, content

    # 3. LM Studio synthetic reasoning marker
    end_pattern = r"__LM_STUDIO_INTERNAL_LSEP_SYNTHETIC_REASONING_END_[a-f0-9]+__"
    matches = list(re.finditer(end_pattern, text))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        thought = text[:start].strip()
        content = text[end:].strip()
        return thought if thought else None, content

    return None, text


def extract_agentic_update(reply: str) -> Optional[str]:
    """
    Extracts structured update_value string or JSON payload from conversational LLM chat responses.
    
    Example:
        extract_agentic_update('```json\n{"update_value": "Phage T4"}\n```') -> "Phage T4"
    """
    if not reply:
        return None

    update_match = re.search(r'```json\s*\{\s*"update_value"\s*:\s*(.*?)\s*\}\s*```', reply, re.DOTALL | re.IGNORECASE)
    if not update_match:
        update_match = re.search(r'\{\s*"update_value"\s*:\s*([^}]+)\s*\}', reply, re.DOTALL | re.IGNORECASE)

    if not update_match:
        return None

    raw_val = update_match.group(1).strip()
    updated_value = None
    try:
        import json
        parsed = json.loads(raw_val)
        if isinstance(parsed, list):
            if len(parsed) > 1:
                updated_value = "\n".join([f"• {v.strip()}" for v in parsed if v and str(v).strip()])
            elif parsed:
                updated_value = str(parsed[0]).strip()
        elif parsed is not None:
            updated_value = str(parsed).strip()
    except Exception:
        clean_str = raw_val.strip('"\' ')
        if clean_str:
            updated_value = clean_str

    if updated_value and updated_value.lower() not in ("null", "none", "false", "", "not specified"):
        return updated_value

    return None

