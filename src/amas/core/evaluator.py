import logging
from typing import Optional
from amas.interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_PROMPT_TEMPLATE = """You are an expert LLM quality assessor. 
Your task is to compare a predicted result against its ground truth and assign a holistic quality score.

**Scoring Scale:**
*   **0:** Complete hallucination or completely incorrect.
*   **10:** Picture-perfect accuracy.

**Input Data:**
---
### Predicted Result
<document>
{prediction}
</document>

### Ground Truth
<document>
{ground_truth}
</document>
---

**Instructions for Output:**
Analyze each key-value of the Predicted Result against the ground truth and generate a single result formatted strictly as a Markdown table. 
Do not include any introductory or concluding sentences outside of the table.

**Required Output Format (MUST follow this structure):**
* Overall score: [0 - 10]

| Key | Prediction | Truth | Score [0 - 10] | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| [Key] | [prediction] | [ground_truth] | [Calculated Score] | [Your detailed analysis of the discrepancy] |
"""

from amas.utils.text import split_llm_response

def clean_llm_response(message: str) -> str:
    """
    Cleans special thought or channel blocks from local LLM response messages.
    """
    _, response = split_llm_response(message)
    return response

class Evaluator:
    """
    Evaluates LLM predictions against ground truth values.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def evaluate(
        self, 
        prediction: str, 
        ground_truth: str, 
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Runs quality assessment comparing prediction text to ground truth using the LLM provider.
        """
        template = prompt_template or DEFAULT_COMPARISON_PROMPT_TEMPLATE
        prompt = template.format(prediction=prediction, ground_truth=ground_truth)
        
        logger.info("Sending comparison assessment to LLM...")
        raw_response = self.llm_provider.respond(prompt, **kwargs)
        
        cleaned_response = clean_llm_response(raw_response)
        return cleaned_response
