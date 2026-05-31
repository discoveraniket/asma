import logging
from typing import Dict, Any, Union, List, Callable
from bioc.biocjson.decoder import parse_collection
from asma.utils.text import clean_text, clean_text_keep_citations, extract_metadata_llm
from asma.utils.xml_table import parse_xml_table_to_csv, parse_xml_table_to_markdown

logger = logging.getLogger(__name__)

def _parse_bioc_to_markdown_base(
    bioc_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    clean_text_fn: Callable[[str], str],
    table_parser_fn: Callable[[str], str]
) -> str:
    """
    Core parsing engine logic. Decodes BioC and loops over passages using callbacks for styling.
    """
    # BioC API response can be a list enclosing the collection dictionary
    obj = bioc_data[0] if isinstance(bioc_data, list) else bioc_data
    
    logger.debug("Decoding BioC collection...")
    collection = parse_collection(obj)
    if not collection.documents:
        logger.warning("No documents found in BioC collection.")
        return ""
        
    doc = collection.documents[0]
    logger.debug(f"Processing BioC document: {doc.id}")
    
    output_lines = []

    for p in doc.passages:
        pt = p.infons.get("type", "")
        st = p.infons.get("section_type", "")
        text = p.text.strip() if p.text else ""
        
        if not text:
            # Table might not have direct text but has xml in infons
            if pt == "table":
                xml_str = p.infons.get("xml", "")
                if xml_str:
                    csv_data = table_parser_fn(xml_str)
                    if csv_data:
                        output_lines.append(csv_data + "\n")
            continue
            
        # Front Matter & Metadata
        if pt == "front":
            output_lines.append(f"# {text}\n")
            meta = extract_metadata_llm(p.infons)
            if meta:
                output_lines.extend(meta)
                output_lines.append("")
            continue
            
        # Skip Reference Section
        if st == "REF" or (pt in ("title", "title_1") and st == "REF") or pt == "ref":
            continue
            
        # Headings
        if pt in ("abstract_title_1", "title_1"):
            output_lines.append(f"## {text}\n")
            continue
        if pt == "title_2":
            output_lines.append(f"### {text}\n")
            continue
        if pt == "title_3":
            output_lines.append(f"#### {text}\n")
            continue
            
        # Figure & Table Captions
        if pt == "fig_caption":
            fig_id = p.infons.get("id", "")
            fig_num = fig_id.replace("fig", "") if fig_id.startswith("fig") else fig_id
            label = f"Figure {fig_num}" if fig_num else "Figure"
            output_lines.append(f"{label}: {clean_text_fn(text)}\n")
            continue
        if pt == "table_caption":
            tab_id = p.infons.get("id", "")
            tab_num = tab_id.replace("tab", "") if tab_id.startswith("tab") else tab_id
            label = f"Table {tab_num}" if tab_num else "Table"
            output_lines.append(f"{label}: {clean_text_fn(text)}\n")
            continue
            
        # Tables (XML to CSV/Markdown conversion)
        if pt == "table":
            xml_str = p.infons.get("xml", "")
            if xml_str:
                csv_data = table_parser_fn(xml_str)
                if csv_data:
                    output_lines.append(csv_data + "\n")
            else:
                output_lines.append(clean_text_fn(text) + "\n")
            continue
            
        # Fallback to general paragraph parsing
        output_lines.append(clean_text_fn(text) + "\n")

    return "\n".join(output_lines)

def parse_bioc_to_llm_markdown(
    bioc_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    clean_text_fn: Callable[[str], str] = clean_text,
    table_parser_fn: Callable[[str], str] = parse_xml_table_to_csv
) -> str:
    """
    Parses BioC JSON collection structure and converts it into a clean,
    LLM-optimized Markdown representation.

    Args:
        bioc_data: The raw BioC collection JSON dictionary or list.
        clean_text_fn: Callback function to clean and normalize passage text.
                       Defaults to `asma.utils.text.clean_text`.
        table_parser_fn: Callback function to convert XML tables to CSV strings.
                         Defaults to `asma.utils.xml_table.parse_xml_table_to_csv`.

    Returns:
        The clean Markdown string representation of the document.
    """
    return _parse_bioc_to_markdown_base(
        bioc_data=bioc_data,
        clean_text_fn=clean_text_fn,
        table_parser_fn=table_parser_fn
    )

def parse_bioc_to_human_markdown(
    bioc_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    clean_text_fn: Callable[[str], str] = clean_text_keep_citations,
    table_parser_fn: Callable[[str], str] = parse_xml_table_to_markdown
) -> str:
    """
    Parses BioC JSON collection structure and converts it into a beautiful,
    accurate, human-friendly Markdown representation.

    Args:
        bioc_data: The raw BioC collection JSON dictionary or list.
        clean_text_fn: Callback function to clean and normalize passage text.
                       Defaults to `asma.utils.text.clean_text_keep_citations`.
        table_parser_fn: Callback function to convert XML tables to Markdown table strings.
                         Defaults to `asma.utils.xml_table.parse_xml_table_to_markdown`.

    Returns:
        The beautiful Markdown string representation of the document.
    """
    return _parse_bioc_to_markdown_base(
        bioc_data=bioc_data,
        clean_text_fn=clean_text_fn,
        table_parser_fn=table_parser_fn
    )
