# Walkthrough: Modularizing the `amas` Package

We have successfully modularized the monolithic [unified_pipeline.py](file:///d:/Codes/GitHub/amas/Modularize_a1/unified_pipeline.py) into an extensible, industry-standard Python library under the name **`amas`** (**Article Mining & Analysis Structurer**).

---

## 1. What Was Created

We restructured the codebase using a `src/` layout configuration.

### Configuration and Build
* [pyproject.toml](file:///d:/Codes/GitHub/amas/Modularize_a1/pyproject.toml): Modern PEP-621 build configuration exposing dependencies (`crossrefapi`, `backoff`, `pymupdf`, `requests`, `bioc`, `beautifulsoup4`, `lmstudio`, `tqdm`, `lxml`) and optional developer packages (like `pytest` and `responses`).

### Interfaces (`src/amas/interfaces/`)
Created abstract base class templates defining components so they are modular and easy to replace in the future:
* [resolver.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/interfaces/resolver.py): The contract for DOI validation and metadata lookup.
* [fetcher.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/interfaces/fetcher.py): The contract for fetching scientific documents.
* [llm.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/interfaces/llm.py): The contract for running inference.

### Providers (`src/amas/providers/`)
Concrete implementations fulfilling the interfaces:
* [resolver_crossref.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/providers/resolver_crossref.py): Resolves DOIs via Crossref.
* [fetcher_pmc.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/providers/fetcher_pmc.py): Uses the NCBI `idconv` and `BioC` RESTful APIs to fetch paper formats, decorated with exponential backoff retry.
* [llm_lmstudio.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/providers/llm_lmstudio.py): Connects to LM Studio via the websocket sync API.

### Core & Utilities (`src/amas/core/` & `src/amas/utils/`)
* [parser.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/core/parser.py): Pure parser logic to extract BioC collections into clean Markdown.
* [evaluator.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/core/evaluator.py): Pure evaluator logic to compare LLM extractions against ground truth and produce comparison reports.
* [text.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/utils/text.py) / [xml_table.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/utils/xml_table.py): Text sanitization and XML table parsers.

### Package Entrypoints & Runner
* [__init__.py](file:///d:/Codes/GitHub/amas/Modularize_a1/src/amas/__init__.py): Exposes clean import points.
* [run_pipeline.py](file:///d:/Codes/GitHub/amas/Modularize_a1/run_pipeline.py): Orchestrates the entire pipeline using the package modules, replicating the exact execution of the original script.

---

## 2. Verification Results

* 28/28 unit tests passed successfully.

### Developer Documentation
* Created [amas_documentation.md](file:///d:/Codes/GitHub/amas/Modularize_a1/amas_documentation.md) which contains a comprehensive guide detailing installation, prerequisites, quick start examples, modular design breakdowns, and extension tutorials.

### User Notebook Migration
* Confirmed that [pipeline.ipynb](file:///d:/Codes/GitHub/amas/Modularize_a1/pipeline.ipynb) is fully migrated to use the modular `amas` API calls (such as `extract_doi_from_pdf`, `PmcFetcher`, and `LMStudioProvider`), replacing all legacy monolithic blocks.

### End-to-End Pipeline Execution
We executed [run_pipeline.py](file:///d:/Codes/GitHub/amas/Modularize_a1/run_pipeline.py) end-to-end:
1. Read `pdf/36374021.pdf`.
2. Extracted and validated DOI (`10.1128/spectrum.01994-22`) successfully via Crossref.
3. Translated DOI using NCBI ID converter API. (Succeeded on retry after hitting a 429 rate limit using exponential backoff!).
4. Fetched the PMC BioC JSON for the resolved PMCID `PMC9769620`.
5. Decoded and parsed XML and tables to Markdown.
6. Loaded LM Studio local server and completed gemma-4 model structured variable extraction.
7. Cleaned and saved results to `./md/36374021_result.md`.
