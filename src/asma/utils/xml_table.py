import csv
import io
import re
from bs4 import BeautifulSoup

def parse_xml_table_to_csv(xml_str: str) -> str:
    """
    Parses an XML string representing a table and converts it into a CSV formatted string.
    """
    soup = BeautifulSoup(xml_str, "lxml-xml")
    table = soup.find("table")
    if not table:
        soup = BeautifulSoup(f"<table>{xml_str}</table>", "lxml-xml")
        table = soup.find("table")
    if not table:
        return ""

    inline_tags = {"sup", "sub", "italic", "bold", "xref"}
    
    def cell_text(cell):
        parts = []
        for child in cell.descendants:
            if isinstance(child, str):
                parts.append(child)
            elif child.name not in inline_tags:
                parts.append(" ")
        raw = "".join(parts)
        return re.sub(r"\s{2,}", " ", raw).strip()

    rows = []
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            rows.append([cell_text(c) for c in header_row.find_all(["th", "td"])])

    if not rows:
        first_tr = table.find("tr")
        if first_tr:
            rows.append([cell_text(c) for c in first_tr.find_all(["th", "td"])])

    if not rows:
        return ""

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    
    for tr in body_rows:
        cells = [cell_text(c) for c in tr.find_all(["td", "th"])]
        rows.append(cells)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().strip()

def parse_xml_table_to_markdown(xml_str: str) -> str:
    """
    Parses an XML string representing a table and converts it into a Markdown table string.
    """
    soup = BeautifulSoup(xml_str, "lxml-xml")
    table = soup.find("table")
    if not table:
        soup = BeautifulSoup(f"<table>{xml_str}</table>", "lxml-xml")
        table = soup.find("table")
    if not table:
        return ""

    inline_tags = {"sup", "sub", "italic", "bold", "xref"}
    
    def cell_text(cell):
        parts = []
        for child in cell.descendants:
            if isinstance(child, str):
                parts.append(child)
            elif child.name not in inline_tags:
                parts.append(" ")
        raw = "".join(parts)
        # Escape pipes in markdown cell content to preserve table formatting
        escaped = re.sub(r"\|", "\\|", raw)
        return re.sub(r"\s{2,}", " ", escaped).strip()

    rows = []
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            rows.append([cell_text(c) for c in header_row.find_all(["th", "td"])])

    if not rows:
        first_tr = table.find("tr")
        if first_tr:
            rows.append([cell_text(c) for c in first_tr.find_all(["th", "td"])])

    if not rows:
        return ""

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    
    for tr in body_rows:
        cells = [cell_text(c) for c in tr.find_all(["td", "th"])]
        rows.append(cells)

    # Convert rows to Markdown table representation
    headers = rows[0]
    markdown_lines = []
    
    # Header line
    markdown_lines.append("| " + " | ".join(headers) + " |")
    # Divider line
    markdown_lines.append("| " + " | ".join([":---"] * len(headers)) + " |")
    
    # Body lines
    for row in rows[1:]:
        # Normalize column count to match headers
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        markdown_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(markdown_lines)
