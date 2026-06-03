import logging
from typing import Dict, Any, Optional
from asma.providers.resolver_crossref import CrossrefResolver, format_crossref_authors
from asma.providers.fetcher_pmc import PmcFetcher
from asma.core.parser import parse_bioc_to_human_markdown, parse_bioc_to_llm_markdown

logger = logging.getLogger(__name__)

class DocumentIngester:
    """
    Orchestrates the metadata resolution, full-text retrieval, and parsing of academic articles.
    """
    def __init__(self, ncbi_email: str = "asma@example.com", cache_dir: Optional[str] = None):
        self.resolver = CrossrefResolver(cache_dir=cache_dir)
        self.fetcher = PmcFetcher(email=ncbi_email, cache_dir=cache_dir)

    def ingest_doi(self, doi: str) -> Dict[str, Any]:
        """
        Takes a DOI, resolves it via Crossref, fetches the PMC full text,
        and parses it to both human-friendly and LLM-friendly Markdown.
        
        Args:
            doi: The DOI of the paper.
            
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

        # 2. Fetch PMC XML/JSON Full Text
        try:
            bioc_data = self.fetcher.fetch_by_doi(doi)
        except Exception as e:
            logger.error(f"NCBI PMC fetch failure for DOI {doi}: {e}")
            raise RuntimeError(f"PMC article extraction failed: {e}")

        # 3. Parse BioC format into Markdown structures
        clean_content = parse_bioc_to_human_markdown(bioc_data)
        raw_content = parse_bioc_to_llm_markdown(bioc_data)

        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "journal": journal,
            "cleanContent": clean_content,
            "rawContent": raw_content
        }
