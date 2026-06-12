import pytest
from unittest.mock import MagicMock
from asma.core.structurer import extract_json_array, structure_extractions

def test_extract_json_array_raw():
    raw_json = '[{"field1": "val1", "field2": "val2"}]'
    res = extract_json_array(raw_json)
    assert len(res) == 1
    assert res[0]["field1"] == "val1"

def test_extract_json_array_markdown():
    md_json = """
    ```json
    [
        {"field1": "val1", "field2": "val2"},
        {"field1": "val3", "field2": "val4"}
    ]
    ```
    """
    res = extract_json_array(md_json)
    assert len(res) == 2
    assert res[0]["field1"] == "val1"
    assert res[1]["field2"] == "val4"

def test_extract_json_array_invalid():
    with pytest.raises(ValueError):
        extract_json_array("not a json array")

    with pytest.raises(ValueError):
        extract_json_array('{"key": "value"}') # object, not array

def test_structure_extractions_success():
    mock_llm = MagicMock()
    mock_response = """
    <|channel>thought
    Let me think... I need to output a JSON array of objects.
    <channel|>
    [
        {"Phage": "Phage 4141", "Latent period": "15"},
        {"Phage": "Phage MJW", "Latent period": "12"}
    ]
    """
    mock_llm.respond.return_value = mock_response

    extractions = {
        "Phage": "Vibrio phage 4141: 15 min; Vibrio phage MJW: 12 min",
        "Latent period": "15 min and 12 min"
    }
    fields = ["Phage", "Latent period", "Optimal MOI"]

    records = structure_extractions(extractions, fields, mock_llm)
    
    assert len(records) == 2
    assert records[0]["Phage"] == "Phage 4141"
    assert records[0]["Latent period"] == "15"
    assert records[0]["Optimal MOI"] == "" # filled with empty string since it was missing
    
    assert records[1]["Phage"] == "Phage MJW"
    assert records[1]["Latent period"] == "12"
    assert records[1]["Optimal MOI"] == ""

    mock_llm.respond.assert_called_once()

def test_structure_extractions_fallback():
    mock_llm = MagicMock()
    mock_llm.respond.side_effect = Exception("LLM connection timed out")

    extractions = {
        "Phage": "Vibrio phage 4141",
        "Latent period": "15"
    }
    fields = ["Phage", "Latent period", "Optimal MOI"]

    # Should not crash, but fallback to single dictionary containing original values
    records = structure_extractions(extractions, fields, mock_llm)
    
    assert len(records) == 1
    assert records[0]["Phage"] == "Vibrio phage 4141"
    assert records[0]["Latent period"] == "15"
    assert records[0]["Optimal MOI"] == ""
