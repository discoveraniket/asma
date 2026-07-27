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
        doi = (doi or "").strip()
        if not doi and not pdf_path:
            raise ValueError("Both DOI and pdf_path are empty.")

        # 1. Resolve DOI metadata (with graceful offline fallback)
        title = "Unknown Title"
        authors = "Local Document"
        journal = "Local PDF Document"

        if doi:
            try:
                doi_meta = self.resolver.resolve_doi(doi)
                if doi_meta:
                    title = doi_meta.get('title', [None])[0] or title
                    authors_list = doi_meta.get('author', [])
                    authors = format_crossref_authors(authors_list)
                    journal = doi_meta.get('container-title', [None])[0] or journal

                    pub_date = doi_meta.get('published-print') or doi_meta.get('created')
                    if pub_date:
                        date_parts = pub_date.get('date-parts', [[None]])
                        year = date_parts[0][0]
                        if year:
                            authors = f"{authors} ({year})"
            except Exception as meta_err:
                logger.warning(f"Crossref DOI resolution failed for {doi} (possibly offline): {meta_err}")

        # Fallback title from PDF file name if Crossref did not resolve a title
        if (title == "Unknown Title" or not title) and pdf_path:
            pdf_file = Path(pdf_path)
            if pdf_file.exists():
                title = pdf_file.stem.replace('_', ' ').replace('-', ' ').title()

        clean_content = None
        raw_content = None

        # 2. Try PMC online XML fetch first if DOI is present
        if doi:
            try:
                bioc_data = self.fetcher.fetch_by_doi(doi)
                if bioc_data:
                    clean_content = parse_bioc_to_human_markdown(bioc_data)
                    raw_content = parse_bioc_to_llm_markdown(bioc_data)
            except Exception as e:
                logger.warning(f"NCBI PMC fetch failure for DOI {doi} (possibly offline): {e}")

        # 3. Fallback to local PDF extraction if PMC didn't return text (e.g. offline or not on PMC)
        if (not clean_content or not raw_content) and pdf_path:
            pdf_file = Path(pdf_path)
            if pdf_file.exists():
                # Primary: pymupdf4llm
                try:
                    import pymupdf4llm
                    logger.info(f"Extracting local PDF content using pymupdf4llm: {pdf_path}")
                    pdf_md = pymupdf4llm.to_markdown(str(pdf_file))
                    if pdf_md and pdf_md.strip():
                        clean_content = pdf_md.strip()
                        raw_content = pdf_md.strip()
                except Exception as pdf_err:
                    logger.error(f"pymupdf4llm extraction failed for {pdf_path}: {pdf_err}")

                # Secondary: PyMuPDF fitz raw text fallback
                if not clean_content or not raw_content:
                    try:
                        import fitz
                        logger.info(f"Extracting local PDF content using fitz text fallback: {pdf_path}")
                        doc = fitz.open(str(pdf_file))
                        pages_text = [page.get_text() for page in doc]
                        full_text = "\n\n".join(t for t in pages_text if t and t.strip())
                        if full_text and full_text.strip():
                            clean_content = full_text.strip()
                            raw_content = full_text.strip()
                    except Exception as fitz_err:
                        logger.error(f"fitz text extraction failed for {pdf_path}: {fitz_err}")

        # Final failsafe: If text is still empty and pdf_path is provided (e.g. scanned image PDF), return placeholder
        if (not clean_content or not raw_content) and pdf_path:
            clean_content = "# Document Content\n\nNo selectable digital text found in this PDF document. It may be a scanned image or protected PDF."
            raw_content = clean_content

        if not clean_content or not raw_content:
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
