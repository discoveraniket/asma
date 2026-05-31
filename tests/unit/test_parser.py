import pytest
from unittest.mock import MagicMock
from asma.core.parser import parse_bioc_to_llm_markdown
from asma.core.evaluator import Evaluator, clean_llm_response

def test_clean_llm_response():
    msg = "<|channel>thought\nSome thoughts here\n<channel|>\nActual Response content"
    assert clean_llm_response(msg) == "Actual Response content"
    
    msg_no_thought = "Only response"
    assert clean_llm_response(msg_no_thought) == "Only response"

def test_parse_bioc_to_markdown_simple():
    # Construct a minimal BioC JSON-like dict to test the parsing pipeline
    bioc_data = {
        "source": "PMC",
        "date": "2026-05-30",
        "key": "pmc.key",
        "infons": {},
        "documents": [
            {
                "id": "12345",
                "infons": {},
                "annotations": [],
                "relations": [],
                "passages": [
                    {
                        "infons": {"type": "front", "article-id_doi": "10.1000/xyz"},
                        "offset": 0,
                        "text": "My Article Title",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    },
                    {
                        "infons": {"type": "title_1"},
                        "offset": 50,
                        "text": "Abstract",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    },
                    {
                        "infons": {"type": "paragraph"},
                        "offset": 100,
                        "text": "This is a body paragraph [1].",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    }
                ]
            }
        ]
    }
    
    markdown = parse_bioc_to_llm_markdown(bioc_data)
    
    assert "# My Article Title" in markdown
    assert "DOI: https://doi.org/10.1000/xyz" in markdown
    assert "## Abstract" in markdown
    assert "This is a body paragraph ." in markdown  # brackets citation cleaned

def test_evaluator():
    mock_llm = MagicMock()
    mock_llm.respond.return_value = "<|channel>thought\nThinking\n<channel|>\nOverall score: 9"
    
    evaluator = Evaluator(mock_llm)
    res = evaluator.evaluate("Prediction data", "Ground truth data", temperature=0.5)
    
    assert res == "Overall score: 9"
    mock_llm.respond.assert_called_once()
    args, kwargs = mock_llm.respond.call_args
    assert "Prediction data" in args[0]
    assert "Ground truth data" in args[0]
    assert kwargs["temperature"] == 0.5

def test_parse_bioc_to_markdown_custom_callbacks():
    bioc_data = {
        "source": "PMC",
        "date": "2026-05-30",
        "key": "pmc.key",
        "infons": {},
        "documents": [
            {
                "id": "12345",
                "infons": {},
                "annotations": [],
                "relations": [],
                "passages": [
                    {
                        "infons": {"type": "paragraph"},
                        "offset": 0,
                        "text": "This is raw text.",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    }
                ]
            }
        ]
    }
    
    # Custom callbacks
    custom_clean = lambda t: t.upper()
    
    markdown = parse_bioc_to_llm_markdown(bioc_data, clean_text_fn=custom_clean)
    assert "THIS IS RAW TEXT." in markdown

def test_parse_bioc_to_human_markdown():
    from asma.core.parser import parse_bioc_to_human_markdown
    bioc_data = {
        "source": "PMC",
        "date": "2026-05-30",
        "key": "pmc.key",
        "infons": {},
        "documents": [
            {
                "id": "12345",
                "infons": {},
                "annotations": [],
                "relations": [],
                "passages": [
                    {
                        "infons": {"type": "paragraph"},
                        "offset": 0,
                        "text": "This is raw text with citation [1].",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    },
                    {
                        "infons": {"type": "table", "xml": "<table><tr><th>H1</th></tr><tr><td>D1</td></tr></table>"},
                        "offset": 100,
                        "text": "",
                        "sentences": [],
                        "annotations": [],
                        "relations": []
                    }
                ]
            }
        ]
    }
    
    markdown = parse_bioc_to_human_markdown(bioc_data)
    # Check that citation is preserved
    assert "citation [1]." in markdown
    # Check that table is parsed into a Markdown table
    assert "| H1 |" in markdown
    assert "| :--- |" in markdown
    assert "| D1 |" in markdown

def test_asma_config_build_prompt():
    from asma.config import AsmaConfig
    config = AsmaConfig()
    
    # Test default prompt construction
    prompt = config.build_prompt(document="Mock text document content.")
    assert "You are an expert data extraction assistant." in prompt
    assert "Primary targeted bacteria species:" in prompt
    assert "Mock text document content." in prompt

    # Test custom fields list override
    custom_fields = ["Bacteria name:", "Optimal pH value:"]
    custom_prompt = config.build_prompt(document="Mock text.", fields=custom_fields)
    assert "You are an expert data extraction assistant." in custom_prompt
    assert "* Bacteria name:" in custom_prompt
    assert "* Optimal pH value:" in custom_prompt
    assert "Primary targeted bacteria species:" not in custom_prompt
    assert "Mock text." in custom_prompt


