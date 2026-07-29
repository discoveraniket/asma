import pytest
from asma.core.cell_extractor import (
    build_extraction_prompt,
    split_llm_response,
    extract_json_array
)

def test_build_extraction_prompt_default():
    prompt = build_extraction_prompt(
        target_column="Latent Period (min)",
        conditions=["the 'Phage' is 'Phage A'", "the 'Target Host' is 'Strain X'"],
        example_val="15"
    )
    
    assert "**QUESTION**: Find the best value/values for 'Latent Period (min)'" in prompt
    assert "When the 'Phage' is 'Phage A'\nand \nthe 'Target Host' is 'Strain X'" in prompt
    assert "- Example style reference (do NOT use this value, only match its format/style): '15'" in prompt


def test_build_extraction_prompt_custom_template():
    custom_template = "Target is {target_column}. Conditions: {conditions}. Example: {example_format}"
    prompt = build_extraction_prompt(
        target_column="MOI",
        conditions=["Phage = A"],
        example_val="0.1",
        template_str=custom_template
    )
    
    assert prompt == "Target is MOI. Conditions: When Phage = A. Example: - Example style reference (do NOT use this value, only match its format/style): '0.1'"


def test_split_llm_response_with_tags():
    llm_output = """
    <|channel>thought
    I need to find the latent period. The text says it is 15 minutes.
    <channel|>
    ["15"]
    """
    thought, content = split_llm_response(llm_output)
    assert thought == "I need to find the latent period. The text says it is 15 minutes."
    assert content == '["15"]'


def test_split_llm_response_no_tags():
    llm_output = '["15"]'
    thought, content = split_llm_response(llm_output)
    assert not thought
    assert content == '["15"]'


def test_extract_json_array_valid():
    res = extract_json_array('["MAK757", "N16961"]')
    assert res == ["MAK757", "N16961"]


def test_extract_json_array_markdown():
    res = extract_json_array("""
    ```json
    ["MAK757", "N16961"]
    ```
    """)
    assert res == ["MAK757", "N16961"]


def test_extract_json_array_literal_eval_fallback():
    # Test with single quotes which standard JSON parser fails on, but ast.literal_eval handles
    res = extract_json_array("['MAK757', 'N16961']")
    assert res == ["MAK757", "N16961"]


def test_extract_json_array_invalid():
    with pytest.raises(ValueError):
        extract_json_array("not a list")
    
    with pytest.raises(ValueError):
        extract_json_array('{"key": "value"}')
