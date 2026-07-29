# `asma` Python Engine Documentation

**`asma`** (**Automated System for Mining Articles**) is a lightweight, modular, and developer-friendly Python engine and library designed to transform unstructured scientific manuscripts (PDFs, BioC XML, Markdown) into structured, queryable data using NCBI/PubMed APIs, Crossref, and LLMs (Google Gemini API & local LM Studio).

> [!NOTE]
> **Scope Boundary**: `asma` is a **headless Python library**. It contains no web server, database, or UI components. Higher-level applications (such as the **`litsift`** desktop/web workspace) import `asma` to perform core document parsing, LLM extraction, and sentence grounding.

---

## Table of Contents
1. [Installation & Requirements](#1-installation--requirements)
2. [LLM Provider Prerequisites (Gemini & LM Studio)](#2-llm-provider-prerequisites-gemini--lm-studio)
3. [Quick Start Example](#3-quick-start-example)
4. [API Reference & Core Modules](#4-api-reference--core-modules)
   - [Document Ingestion & Parsing (`asma.core.ingester`)](#document-ingestion--parsing-asmacoreingester)
   - [Document Processing & DOI Extraction (`asma.utils.document`)](#document-processing--doi-extraction-asmautilsdocument)
   - [DOI Validation & Resolution (`asma.providers.resolver_crossref`)](#doi-validation--resolution-asmaprovidersresolver_crossref)
   - [Article Fetching (`asma.providers.fetcher_pmc`)](#article-fetching-asmaprovidersfetcher_pmc)
   - [BioC-to-Markdown Parsers (`asma.core.parser`)](#bioc-to-markdown-parsers-asmacoreparser)
   - [Cell Extraction & Prompting (`asma.core.cell_extractor`)](#cell-extraction--prompting-asmacorecell_extractor)
   - [LLM Inference Providers (`asma.providers`)](#llm-inference-providers-asmaproviders)
   - [Source Sentence Alignment & Grounding (`asma.utils.alignment`)](#source-sentence-alignment--grounding-asmautilsalignment)
   - [Evaluation Framework (`asma.core.evaluator`)](#evaluation-framework-asmacoreevaluator)
   - [Text Cleaning & Tag Splitting (`asma.utils.text`)](#text-cleaning--tag-splitting-asmautilstext)
5. [Advanced Customization & Extension](#5-advanced-customization--extension)

---

## 1. Installation & Requirements

Install the core `asma` package in editable mode:
```bash
pip install -e .
```

### Optional Extras (PDF Support)
To extract DOIs directly from local PDF files, you need to install the `pdf` extra dependency (which installs `PyMuPDF` under the hood):
```bash
pip install asma[pdf]
```

### Dependency Configuration
All dependencies are declared and managed under PEP-621 standards in the `pyproject.toml` file:
* **Core dependencies:** `crossrefapi`, `backoff`, `requests`, `bioc`, `beautifulsoup4`, `lmstudio`, `tqdm`, `lxml`.
* **Optional extra (`pdf`):** `pymupdf`.
* **Development extras (`dev`):** `pytest`, `pytest-cov`, `responses` (for mock unit tests).

---

## 2. LLM Provider Prerequisites (Gemini & LM Studio)

`asma` supports both cloud-based LLM APIs and offline local models:

### A. Google Gemini API Provider (`GeminiProvider`)
Set your Google API Key as an environment variable or pass it directly:
```bash
set GEMINI_API_KEY=your_api_key_here
```

### B. LM Studio Provider (`LMStudioProvider`)
For local, offline inference:
1. Open **LM Studio**.
2. Select and load your model (e.g. `google/gemma-4-e2b`).
3. Start the **Local Inference Server** (`http://127.0.0.1:1234`).
4. Set context length to `16000`+ tokens in LM Studio hardware settings.

---

## 3. Quick Start Example

Here is a complete example of running document ingestion, extraction, and source sentence grounding:

```python
from asma import (
    DocumentIngester,
    GeminiProvider,
    build_extraction_prompt,
    extract_json_array,
    find_source_sentences
)

# 1. Ingest PDF, Markdown, or XML file
ingester = DocumentIngester()
doc_text = ingester.ingest_file("paper.pdf")

# 2. Build extraction prompt for target fields
target_fields = [
    {"name": "organism", "description": "Target organism studied"},
    {"name": "sample_size", "description": "Number of participants/samples"}
]
prompt = build_extraction_prompt(doc_text, target_fields)

# 3. Initialize Gemini LLM Provider
llm = GeminiProvider(model_name="gemini-2.5-flash")
response_text = llm.respond(prompt)

# 4. Extract structured JSON results
extractions = extract_json_array(response_text)
print("Extracted fields:", extractions)

# 5. Ground assertions to verbatim source text
for item in extractions:
    val = item.get("value")
    if val:
        sources = find_source_sentences(val, doc_text)
        print(f"Grounding sources for '{val}':", sources)
```

---

## 4. API Reference & Core Modules

### Document Processing (`asma.utils.document`)
Exposes local document manipulation helpers.

#### `extract_doi_from_pdf`
Extracts DOI identifiers from local PDF files using lazy-loading (does not load `fitz` into memory until called).
```python
from asma import extract_doi_from_pdf

doi = extract_doi_from_pdf(
    pdf_path="paper.pdf",
    max_pages=3,                    # (Optional) Max pages from startup to scan
    doi_pattern=r'10\.\d{4,9}/...'  # (Optional) Override default regex pattern
)
```

---

### DOI Validation (`asma.utils.doi`)
Performs registry checks on DOIs.

#### `validate_doi`
Validates DOI presence using Crossref or custom lookup resolvers.
```python
from asma import validate_doi

# Validate via default Crossref resolver
is_valid = validate_doi("10.1128/spectrum.01994-22")

# Validate using custom Resolver class
is_valid = validate_doi("10.1128/spectrum.01994-22", resolver=MyCustomResolver())
```

---

### Article Fetching (`asma.providers.fetcher_pmc`)
Fetches BioC JSON formats from PubMed Central (PMC).

#### `PmcFetcher`
Resolves DOIs to PMCID/PMID identifiers via NCBI `idconv` and retrieves full-text BioC JSON structures. Features exponential retry backoff.
```python
from asma import PmcFetcher

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

### BioC-to-Markdown Parsers (`asma.core.parser`)
Converts XML/JSON BioC format into Markdown.

#### `parse_bioc_to_llm_markdown`
Designed specifically for LLM input context. Strips citations (e.g., `[1]`, `[2-4]`) to minimize tokens and converts tables to raw CSV layout (which is easier for LLMs to parse).
```python
from asma import parse_bioc_to_llm_markdown

llm_md = parse_bioc_to_llm_markdown(bioc_data)
```

#### `parse_bioc_to_human_markdown`
Designed for human reading. Preserves citation references and renders XML tables as visually formatted Markdown tables.
```python
from asma import parse_bioc_to_human_markdown

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

### Dynamic Prompt Engineering (`asma.config`)
Decouples prompt instructions from the schema fields to be extracted.

#### `AsmaConfig.build_prompt`
Constructs the prompt by combining global extraction instructions with specific schema lists.
```python
from asma import AsmaConfig

config = AsmaConfig()

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

### LLM Inference Client (`asma.providers.llm_lmstudio`)
Client wrapper communicating with LM Studio.

#### `LMStudioProvider`
Initializes model websocket connections, runs inference, validates token sizes, and controls streaming.
```python
from asma import LMStudioProvider

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

### Evaluation Framework (`asma.core.evaluator`)
Performs double-blind validation checks comparing extraction prediction results to ground truth templates.

#### `Evaluator.evaluate`
Sends predictions and ground truths into a comparison matrix prompt using the provider client.
```python
from asma import Evaluator, LMStudioProvider

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

### Custom Tag Splitting (`asma.utils.text`)
Unpacks reasoning thought processes from final output results.

#### `split_llm_response`
Modular tag parsing splitting. Automatically identifies LM Studio's channel tags, DeepSeek `<think>` tags, and supports custom user overrides.
```python
from asma import split_llm_response

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
from asma import LLMProvider

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
The rest of the `asma` parsing, prompt-building, and validation infrastructure remains exactly the same!
