# `asma` — Automated System for Mining Articles

[![PyPI version](https://img.shields.io/pypi/v/asma.svg)](https://pypi.org/project/asma/)
[![Python versions](https://img.shields.io/pypi/pyversions/asma.svg)](https://pypi.org/project/asma/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`asma` is a modular, extensible Python library designed to automate the ingestion, parsing, and structured PICO extraction of scientific research articles from local PDFs and NCBI PMC APIs using Large Language Models (LLMs).

## Features

- **Automated DOI & Metadata Extraction:** Scans local PDF files to extract DOIs and resolves metadata via Crossref with graceful offline fallback.
- **NCBI PMC & Local PDF Dual Ingestion:**
  - **Online**: Resolves DOIs to PMCIDs and PubMed IDs, fetching BioC XML/JSON full text.
  - **Offline / Local PDF Fallback**: Extracts structured Markdown directly from local PDF files using `pymupdf4llm` with PyMuPDF (`fitz`) fallback when PMC is unavailable or offline.
- **Dual-Purpose BioC & Markdown Parsing:**
  - **LLM-optimized Markdown:** Strips references and serializes tables to raw CSV layout to conserve context windows and improve extraction accuracy.
  - **Human-optimized Markdown:** Preserves references and formats clean Markdown tables for easy reading.
- **LLM Provider Agnostic:** Interface-driven (`LLMProvider`) to easily swap between **Google Gemini API** (`GeminiProvider`), **LM Studio** (`LMStudioProvider`), **Ollama**, or custom backends.
- **Structured PICO Cell Extraction & Evaluation:** Run single-pass or multi-pass extraction of key variables (PICO parameters, host strains, MOI, dosages) with automated ground-truth quality evaluation.

---

## Repository Structure

```text
├── src/
│   └── asma/              # Core library source code
│       ├── core/          # Document Ingester, Cell Extractor, Parser & Evaluator engine
│       ├── interfaces/    # LLMProvider abstract interface
│       ├── providers/     # Crossref resolvers, PMC fetchers, Gemini & LM Studio clients
│       └── utils/         # PDF DOI extractors, XML parsers, text utils
├── tests/                 # Unit test suite (48 tests)
├── docs/                  # Detailed SDK developer documentation
│   └── asma_documentation.md
├── notebooks/             # Research, pipeline orchestrator & experiment notebooks
│   ├── pipeline.ipynb
│   ├── run_pipeline.py
│   ├── demo_extraction.ipynb
│   └── demo_extraction_multipass.ipynb
└── pyproject.toml         # Package definition (PEP-621)
```

---

## Quick Start

### 1. Installation

Install the package directly from PyPI:
```bash
pip install asma
```

To enable local PDF processing and offline Markdown extraction, install with the PDF extra:
```bash
pip install "asma[pdf]"
```

### 2. Basic Usage (Python SDK)

```python
from asma import DocumentIngester, CellExtractor

# Ingest article from local PDF or DOI
ingester = DocumentIngester()
article = ingester.ingest_doi("10.1016/j.cell.2020.01.001", pdf_path="sample.pdf")

print("Title:", article["title"])
print("Clean Markdown:", article["cleanContent"][:300])
```

### 3. Detailed Documentation

For a comprehensive guide, read the [Developer Reference Guide](docs/asma_documentation.md):
- [Installation & Requirements](docs/asma_documentation.md#1-installation--requirements)
- [Quick Start Code Example](docs/asma_documentation.md#3-quick-start-example)
- [API Reference & Core Modules](docs/asma_documentation.md#4-api-reference--core-modules)
- [Advanced Customization & Swapping Providers](docs/asma_documentation.md#5-advanced-customization--future-extension)

---

## Local Development & Contribution

If you want to contribute or build from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/discoveraniket/asma.git
   cd asma
   ```
2. Install in editable mode with development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run the unit test suite:
   ```bash
   pytest
   ```

---

## 👥 Authors & Credits

* **[Aniket Sarkar](https://github.com/discoveraniket)** - Lead Software Architect & Developer.
* **Dr. Adhip Mukhopadhyay** - Co-Author & Virology Domain Specialist (Virology domain guidance, systematic review workflow design, and PICO schema validation)
