import logging
from typing import Dict, Any, Optional
from asma.providers.resolver_crossref import CrossrefResolver, format_crossref_authors
from asma.providers.fetcher_pmc import PmcFetcher
from asma.core.parser import parse_bioc_to_human_markdown, parse_bioc_to_llm_markdown

from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentIngester:
    """
    Orchestrates the metadata resolution, full-text retrieval, and parsing of academic articles.
    """
    def __init__(self, ncbi_email: str = "asma@example.com", cache_dir: Optional[str] = None):
        self.resolver = CrossrefResolver(cache_dir=cache_dir)
        self.fetcher = PmcFetcher(email=ncbi_email, cache_dir=cache_dir)

    def ingest_doi(self, doi: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Takes a DOI, resolves it via Crossref, fetches the PMC full text (or falls back to local PDF via pymupdf4llm),
        and parses it to both human-friendly and LLM-friendly Markdown.
        
        Args:
            doi: The DOI of the paper.
            pdf_path: Optional path to the local PDF file for fallback extraction.
            
        Returns:
            Dict containing title, authors, journal, cleanContent, and rawContent.
        """
        doi = doi.strip()
        if not doi:
            raise ValueError("DOI value is empty.")

        # 1. Resolve DOI
        doi_meta = self.resolver.resolve_doi(doi)
        if doi_meta is None:
            raise ValueError(f"DOI '{doi}' validation failed on Crossref registry.")

        title = doi_meta.get('title', [None])[0] or "Unknown Title"
        authors_list = doi_meta.get('author', [])
        authors = format_crossref_authors(authors_list)
        journal = doi_meta.get('container-title', [None])[0] or "Unknown Journal"

        pub_date = doi_meta.get('published-print') or doi_meta.get('created')
        if pub_date:
            date_parts = pub_date.get('date-parts', [[None]])
            year = date_parts[0][0]
            if year:
                authors = f"{authors} ({year})"

        # 2. Fetch PMC XML/JSON Full Text (with fallback to local PDF via pymupdf4llm)
        clean_content = None
        raw_content = None
        pmc_error = None

        try:
            bioc_data = self.fetcher.fetch_by_doi(doi)
            clean_content = parse_bioc_to_human_markdown(bioc_data)
            raw_content = parse_bioc_to_llm_markdown(bioc_data)
        except Exception as e:
            pmc_error = e
            logger.warning(f"NCBI PMC fetch failure for DOI {doi}: {e}")

        # 3. Fallback to local PDF parsing via pymupdf4llm if PMC failed
        if (clean_content is None or raw_content is None) and pdf_path:
            pdf_file = Path(pdf_path)
            if pdf_file.exists():
                try:
                    import pymupdf4llm
                    logger.info(f"Extracting local PDF content for DOI {doi} using pymupdf4llm: {pdf_path}")
                    pdf_md = pymupdf4llm.to_markdown(str(pdf_file))
                    if pdf_md and pdf_md.strip():
                        clean_content = pdf_md.strip()
                        raw_content = pdf_md.strip()
                except Exception as pdf_err:
                    logger.error(f"pymupdf4llm extraction failed for {pdf_path}: {pdf_err}")

        if clean_content is None or raw_content is None:
            err_msg = f"PMC article extraction failed: {pmc_error}" if pmc_error else "Failed to extract article text."
            raise RuntimeError(err_msg)

        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "journal": journal,
            "cleanContent": clean_content,
            "rawContent": raw_content
        }
