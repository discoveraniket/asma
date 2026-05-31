# `amas` Python SDK Documentation

**`amas`** (**Article Mining & Analysis Structurer**) is a lightweight, modular, and developer-friendly Python library designed to streamline the process of transforming unstructured scientific research manuscripts (PDFs) into structured, queryable data using NCBI/PubMed APIs and Local Large Language Models (LLMs).

---

## Table of Contents
1. [Installation & Requirements](#1-installation--requirements)
2. [Prerequisites (LM Studio Setup)](#2-prerequisites-lm-studio-setup)
3. [Quick Start Example](#3-quick-start-example)
4. [API Reference & Core Modules](#4-api-reference--core-modules)
   - [Document Processing (`amas.utils.document`)](#document-processing-amasutilsdocument)
   - [DOI Validation (`amas.utils.doi`)](#doi-validation-amasutilsdoi)
   - [Article Fetching (`amas.providers.fetcher_pmc`)](#article-fetching-amasprovidersfetcher_pmc)
   - [BioC-to-Markdown Parsers (`amas.core.parser`)](#bioc-to-markdown-parsers-amascoreparser)
   - [Dynamic Prompt Engineering (`amas.config`)](#dynamic-prompt-engineering-amasconfig)
   - [LLM Inference Client (`amas.providers.llm_lmstudio`)](#llm-inference-client-amasprovidersllm_lmstudio)
   - [Evaluation Framework (`amas.core.evaluator`)](#evaluation-framework-amascoreevaluator)
   - [Custom Tag Splitting (`amas.utils.text`)](#custom-tag-splitting-amasutilstext)
5. [Advanced Customization & Future Extension](#5-advanced-customization--future-extension)

---

## 1. Installation & Requirements

Install the core `amas` package in editable mode:
```bash
pip install -e .
```

### Optional Extras (PDF Support)
To extract DOIs directly from local PDF files, you need to install the `pdf` extra dependency (which installs `PyMuPDF` under the hood):
```bash
pip install amas[pdf]
```

### Dependency Configuration
All dependencies are declared and managed under PEP-621 standards in the `pyproject.toml` file:
* **Core dependencies:** `crossrefapi`, `backoff`, `requests`, `bioc`, `beautifulsoup4`, `lmstudio`, `tqdm`, `lxml`.
* **Optional extra (`pdf`):** `pymupdf`.
* **Development extras (`dev`):** `pytest`, `pytest-cov`, `responses` (for mock unit tests).

---

## 2. Prerequisites (LM Studio Setup)

Before running the LLM inference or evaluation steps, you must have **LM Studio** loaded and running locally:
1. Open **LM Studio**.
2. Select your desired reasoning model (e.g., `google/gemma-4-e2b`).
3. Load the model and start the **Local Inference Server**.
4. In the right-hand settings panel under **Hardware Settings**, ensure the **Context Length** is configured to fit your target articles (we recommend `16000` or `24000` tokens for full research papers).

---

## 3. Quick Start Example

Here is a complete, copy-pasteable script showing the entire pipeline running end-to-end on a local PDF file:

```python
import json
from amas import (
    AmasConfig,
    extract_doi_from_pdf,
    validate_doi,
    PmcFetcher,
    parse_bioc_to_llm_markdown,
    LMStudioProvider,
    split_llm_response
)

# 1. Configuration Setup
config = AmasConfig(
    model_name="google/gemma-4-e2b",
    ncbi_email="researcher@example.com"
)

# 2. Extract DOI from PDF
pdf_file = "pdf/36374021.pdf"
doi = extract_doi_from_pdf(pdf_file, max_pages=2)
print(f"Extracted DOI: {doi}")

# 3. Validate DOI via Crossref
if not validate_doi(doi, method="crossref"):
    raise ValueError("Invalid DOI detected on Crossref registry.")

# 4. Fetch BioC JSON from NCBI PMC API
fetcher = PmcFetcher(email=config.ncbi_email)
bioc_data = fetcher.fetch_by_doi(doi)

# 5. Parse JSON to LLM-friendly Markdown
llm_friendly_md = parse_bioc_to_llm_markdown(bioc_data)

# 6. Initialize local LLM client
llm = LMStudioProvider(model_name=config.model_name)

# 7. Dynamically construct extraction prompt
prompt = config.build_prompt(document=llm_friendly_md)

# 8. Run inference (streams ingestion progress & live tokens to console)
raw_response = llm.respond(prompt, temperature=0.1)

# 9. Clean and split reasoning thought block from final answer
thought, answer = split_llm_response(raw_response)

print("\n=== Reasonings ===")
print(thought)
print("\n=== Extracted Schema ===")
print(answer)
```

---

## 4. API Reference & Core Modules

### Document Processing (`amas.utils.document`)
Exposes local document manipulation helpers.

#### `extract_doi_from_pdf`
Extracts DOI identifiers from local PDF files using lazy-loading (does not load `fitz` into memory until called).
```python
from amas import extract_doi_from_pdf

doi = extract_doi_from_pdf(
    pdf_path="paper.pdf",
    max_pages=3,                    # (Optional) Max pages from startup to scan
    doi_pattern=r'10\.\d{4,9}/...'  # (Optional) Override default regex pattern
)
```

---

### DOI Validation (`amas.utils.doi`)
Performs registry checks on DOIs.

#### `validate_doi`
Validates DOI presence using Crossref or custom lookup resolvers.
```python
from amas import validate_doi

# Validate via default Crossref resolver
is_valid = validate_doi("10.1128/spectrum.01994-22")

# Validate using custom Resolver class
is_valid = validate_doi("10.1128/spectrum.01994-22", resolver=MyCustomResolver())
```

---

### Article Fetching (`amas.providers.fetcher_pmc`)
Fetches BioC JSON formats from PubMed Central (PMC).

#### `PmcFetcher`
Resolves DOIs to PMCID/PMID identifiers via NCBI `idconv` and retrieves full-text BioC JSON structures. Features exponential retry backoff.
```python
from amas import PmcFetcher

fetcher = PmcFetcher(
    email="user@domain.com",  # Required for NCBI API usage
    tool="custom_extractor",   # Custom identifier
    max_tries=5               # Max retries on HTTP errors (e.g. 429 Rate Limits)
)

# 1. Resolve DOI to PMID/PMCID mapping
ids = fetcher.resolve_doi_to_ids("10.1128/spectrum.01994-22")
# Returns: {'pmcid': 'PMC9769620', 'pmid': '36374021', ...}

# 2. Get full BioC JSON
bioc_data = fetcher.fetch_by_doi("10.1128/spectrum.01994-22")
```

---

### BioC-to-Markdown Parsers (`amas.core.parser`)
Converts XML/JSON BioC format into Markdown.

#### `parse_bioc_to_llm_markdown`
Designed specifically for LLM input context. Strips citations (e.g., `[1]`, `[2-4]`) to minimize tokens and converts tables to raw CSV layout (which is easier for LLMs to parse).
```python
from amas import parse_bioc_to_llm_markdown

llm_md = parse_bioc_to_llm_markdown(bioc_data)
```

#### `parse_bioc_to_human_markdown`
Designed for human reading. Preserves citation references and renders XML tables as visually formatted Markdown tables.
```python
from amas import parse_bioc_to_human_markdown

human_md = parse_bioc_to_human_markdown(bioc_data)
```

#### Custom Callback Injections (Granular Parser Control)
Both parser wrappers accept override callbacks for text cleaning and table parsing:
```python
# Custom clean handler keeping only uppercase letters
custom_clean = lambda text: text.upper()

custom_md = parse_bioc_to_llm_markdown(
    bioc_data,
    clean_text_fn=custom_clean,
    table_parser_fn=custom_table_parser
)
```

---

### Dynamic Prompt Engineering (`amas.config`)
Decouples prompt instructions from the schema fields to be extracted.

#### `AmasConfig.build_prompt`
Constructs the prompt by combining global extraction instructions with specific schema lists.
```python
from amas import AmasConfig

config = AmasConfig()

# 1. Build prompt with default virology schema fields
prompt = config.build_prompt(document=llm_friendly_md)

# 2. Build prompt with custom on-the-fly fields
my_fields = [
    "Host bacteria:",
    "Optimal pH:",
    "Genome size (bp):"
]
prompt = config.build_prompt(document=llm_friendly_md, fields=my_fields)
```

---

### LLM Inference Client (`amas.providers.llm_lmstudio`)
Client wrapper communicating with LM Studio.

#### `LMStudioProvider`
Initializes model websocket connections, runs inference, validates token sizes, and controls streaming.
```python
from amas import LMStudioProvider

llm = LMStudioProvider(model_name="google/gemma-4-e2b")

# Run inference with custom temperature and generation parameters
raw_response = llm.respond(
    prompt,
    temperature=0.1,         # Low temperature = high accuracy
    max_tokens=1500,         # Limit output length
    stream=True,             # Streams prompt ingestion status & output tokens
    ignore_context_limit=False  # Raises ValueError if token count exceeds window
)
```

---

### Evaluation Framework (`amas.core.evaluator`)
Performs double-blind validation checks comparing extraction prediction results to ground truth templates.

#### `Evaluator.evaluate`
Sends predictions and ground truths into a comparison matrix prompt using the provider client.
```python
from amas import Evaluator, LMStudioProvider

llm = LMStudioProvider(model_name="meta-llama-3-8b")
evaluator = Evaluator(llm_provider=llm)

# Run verification and output comparison markdown table report
comparison_report = evaluator.evaluate(
    prediction=predicted_md,
    ground_truth=ground_truth_md,
    temperature=0.1
)
```

---

### Custom Tag Splitting (`amas.utils.text`)
Unpacks reasoning thought processes from final output results.

#### `split_llm_response`
Modular tag parsing splitting. Automatically identifies LM Studio's channel tags, DeepSeek `<think>` tags, and supports custom user overrides.
```python
from amas import split_llm_response

# Unpack standard tags
thought, response = split_llm_response(raw_output)

# Unpack custom tags
thought, response = split_llm_response(
    raw_output,
    start_tag="[reasoning]",
    end_tag="[/reasoning]"
)
```

---

## 5. Advanced Customization & Future Extension

### Swapping the LLM Provider (e.g. Ollama, OpenAI)
The library is fully interface-driven. To add a new provider like **Ollama**, implement the `LLMProvider` interface class:

```python
from typing import Optional, Callable
from amas import LLMProvider

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def respond(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        progress_callback: Optional[Callable[[float], None]] = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        # 1. Connect to Ollama API endpoint (localhost:11434)
        # 2. Call generation with kwargs (temperature, max_tokens)
        # 3. Stream back tokens if stream=True
        return "Ollama Response Content"
```

Now, replace it directly in your pipeline flow:
```python
llm = OllamaProvider(model_name="gemma2")
result = llm.respond(prompt)
```
The rest of the `amas` parsing, prompt-building, and validation infrastructure remains exactly the same!
