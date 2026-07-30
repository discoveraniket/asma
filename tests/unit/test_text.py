from asma.utils.text import clean_text, extract_metadata_llm

def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("line1\nline2") == "line1 line2"

def test_clean_text_removes_bracketed_citations():
    assert clean_text("This is a statement [1].") == "This is a statement ."
    assert clean_text("Some studies [2, 3] and others [4-6] show this.") == "Some studies and others show this."
    assert clean_text("Multiple digits [123, 456].") == "Multiple digits ."
    assert clean_text("No bracket 123.") == "No bracket 123."

def test_extract_metadata_llm_empty():
    assert extract_metadata_llm({}) == []

def test_extract_metadata_llm_full():
    infons = {
        "article-id_doi": "10.1000/xyz123",
        "article-id_pmc": "PMC99999",
        "article-id_pmid": "88888",
        "volume": "12",
        "issue": "3",
        "year": "2024",
        "name_0": "given-names:John;surname:Doe",
        "name_1": "given-names:Jane;surname:Smith",
        "kwd": "phage, biology"
    }
    expected = [
        "DOI: https://doi.org/10.1000/xyz123",
        "PMC: PMC99999",
        "PMID: 88888",
        "Published: Volume 12, Issue 3, (2024)",
        "Authors: John Doe, Jane Smith",
        "Keywords: phage, biology"
    ]
    assert extract_metadata_llm(infons) == expected

def test_split_llm_response():
    from asma.utils.text import split_llm_response
    
    # Test LM Studio gemma-4 format
    raw_lms = "<|channel>thought\nThinking process\n<channel|>\nActual response output"
    thought, response = split_llm_response(raw_lms)
    assert thought == "Thinking process"
    assert response == "Actual response output"
    
    # Test standard think tag format
    raw_think = "<think>\nThinking inside tags\n</think>\nResponse content here"
    thought, response = split_llm_response(raw_think)
    assert thought == "Thinking inside tags"
    assert response == "Response content here"
    
    # Test custom override
    raw_custom = "[custom_start]Custom thought[custom_end]Custom response"
    thought, response = split_llm_response(raw_custom, start_tag="[custom_start]", end_tag="[custom_end]")
    assert thought == "Custom thought"
    assert response == "Custom response"
    
    # Test fallback when no tags exist
    raw_plain = "Plain text only"
    thought, response = split_llm_response(raw_plain)
    assert thought is None
    assert response == "Plain text only"



def test_generate_anchor_ids():
    from asma import generate_anchor_ids
    anchors = generate_anchor_ids("Phage Name", 3)
    assert anchors == ["phage-name-anchor-1", "phage-name-anchor-2", "phage-name-anchor-3"]

    empty_anchors = generate_anchor_ids("", 1)
    assert empty_anchors == ["anchor-anchor-1"]


def test_extract_agentic_update():
    from asma import extract_agentic_update

    reply_json = 'Some reasoning\n```json\n{\n  "update_value": "Bacteriophage T4"\n}\n```'
    assert extract_agentic_update(reply_json) == "Bacteriophage T4"

    reply_list = '```json\n{\n  "update_value": ["Host A", "Host B"]\n}\n```'
    assert extract_agentic_update(reply_list) == "• Host A\n• Host B"

    reply_invalid = 'No json update block'
    assert extract_agentic_update(reply_invalid) is None

