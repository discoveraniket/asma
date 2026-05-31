# `asma` — Automated System for Mining Articles

[![PyPI version](https://img.shields.io/pypi/v/asma.svg)](https://pypi.org/project/asma/)
[![Python versions](https://img.shields.io/pypi/pyversions/asma.svg)](https://pypi.org/project/asma/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`asma` is a modular, extensible Python library designed to automate the ingestion, parsing, and structured extraction of scientific research articles from PDFs using NCBI PMC APIs and local Large Language Models (LLMs).

## Features

- **Automated DOI Extraction & Validation:** Lazy-loads PDF processing utilities to scan and validate DOIs via Crossref.
- **NCBI PMC Ingestion:** Resolves DOIs to PMCIDs and PubMed IDs with robust exponential backoff.
- **Dual-Purpose BioC Parsing:**
  - **LLM-optimized Markdown:** Strips references and serializes tables to raw CSV layout to conserve tokens and improve accuracy.
  - **Human-optimized Markdown:** Preserves references and builds clean Markdown tables for easy reading.
- **In-Context Prompt Engineering:** Decouples instruction logic from schema definition fields to support dynamic prompting.
- **LLM Provider Agnostic:** Interface-driven (`LLMProvider`) to easily swap between LM Studio, Ollama, OpenAI, or other backends.
- **Automated Validation:** Evaluate extraction outputs against ground-truth files and generate markdown report cards.

---

## Repository Structure

```text
├── src/
│   └── asma/              # Main library source code
│       ├── core/          # Markdown parsing and Evaluation engine
│       ├── providers/     # Crossref resolvers, PMC fetchers, LM Studio client
│       ├── utils/         # PDF helpers, XML parsers, text utils
│       └── config.py      # Prompt templates and default schemas
├── tests/                 # Unit test suite
├── run_pipeline.py        # End-to-end command-line orchestrator
├── pipeline.ipynb         # Interactive Jupyter demo notebook
├── pyproject.toml         # Package definition (PEP-621)
└── asma_documentation.md  # Detailed SDK reference & developer guide
```

---

## Quick Start

### 1. Installation

Install the package directly from PyPI:
```bash
pip install asma
```

To enable local PDF DOI extraction, install with the PDF support extra (which installs `PyMuPDF`):
```bash
pip install "asma[pdf]"
```

### 2. Run the Orchestrator

To run the pipeline end-to-end (requires a local model server loaded on LM Studio):
```bash
python run_pipeline.py 36374021
```

### 3. Detailed Documentation

For a comprehensive guide covering custom schemas, extending providers (like Ollama), streaming controls, and the testing framework, read the [Developer Reference Guide](asma_documentation.md).

---

## Local Development & Contribution

If you want to contribute or build from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/asma.git
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
