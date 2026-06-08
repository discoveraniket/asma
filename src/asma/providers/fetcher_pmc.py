import logging
import requests
import backoff
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from asma.interfaces.fetcher import ArticleFetcher

logger = logging.getLogger(__name__)

class PmcFetcher(ArticleFetcher):
    """
    Fetches BioC JSON from NCBI PMC/PubMed APIs by translating DOIs.
    Supports optional disk caching.
    """
    def __init__(
        self, 
        email: str = "asma@example.com", 
        tool: str = "asma_extractor",
        idconv_base_url: str = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
        bionlp_base_url: str = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful",
        max_tries: int = 3,
        cache_dir: Optional[str] = None
    ):
        self.email = email
        self.tool = tool
        self.idconv_base_url = idconv_base_url
        self.bionlp_base_url = bionlp_base_url
        self.max_tries = max_tries
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _get_cache_path(self, doi: str) -> Path:
        hashed = hashlib.sha256(doi.strip().lower().encode('utf-8')).hexdigest()
        return self.cache_dir / "pmc_fetches" / f"{hashed}.json"

    def _get_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        @backoff.on_exception(
            backoff.expo, 
            requests.exceptions.RequestException, 
            max_tries=self.max_tries
        )
        def _get():
            logger.debug(f"HTTP GET: {url} with params {params}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response

        return _get()

    def resolve_doi_to_ids(self, doi: str) -> Dict[str, Any]:
        """
        Translates a DOI to PMCID and PMID via the NCBI ID Converter API.
        """
        params = {
            "ids": doi,
            "format": "json",
            "tool": self.tool,
            "email": self.email
        }
        try:
            logger.info(f"Resolving DOI {doi} to PMC/PubMed IDs...")
            resp = self._get_with_retry(self.idconv_base_url, params=params)
            data = resp.json()
            records = data.get('records', [{}])
            if not records:
                raise ValueError(f"No ID mapping records found for DOI: {doi}")
            return records[0]
        except Exception as e:
            logger.error(f"Failed to resolve DOI {doi} to IDs: {e}")
            raise ValueError(f"Failed to resolve DOI to PMC/PubMed IDs: {e}") from e

    def fetch_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Resolves the DOI and fetches the BioC JSON.
        Checks cache first if cache_dir is specified.
        """
        doi = doi.strip()
        
        # 1. Check cache first
        if self.cache_dir:
            cache_file = self._get_cache_path(doi)
            if cache_file.exists():
                try:
                    logger.info(f"NCBI PMC fetch retrieved from cache: {doi}")
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read PMC cache file {cache_file}: {e}")

        # 2. Resolve target ID (PMCID/PMID)
        record = self.resolve_doi_to_ids(doi)
        pmcid = record.get('pmcid')
        pmid = record.get('pmid')

        bioc_data = None
        last_error = None

        # Try PMCID full text first if available
        if pmcid:
            bioc_url = f"{self.bionlp_base_url}/pmcoa.cgi/BioC_json/{pmcid}/unicode"
            logger.info(f"Fetching BioC JSON full text from PMC: {bioc_url}")
            try:
                resp = self._get_with_retry(bioc_url)
                # Check for standard API response containing "No result can be found" in a 200 response
                if "[Error]" in resp.text and "No result" in resp.text:
                    logger.warning(f"PMC full text BioC JSON not found for {pmcid} (might not be open access).")
                else:
                    bioc_data = resp.json()
            except Exception as e:
                logger.warning(f"Failed to fetch open access BioC JSON for {pmcid}: {e}")
                last_error = e

        # Fallback to PMID abstract if full text failed or wasn't available
        if not bioc_data and pmid:
            bioc_url = f"{self.bionlp_base_url}/pubmed.cgi/BioC_json/{pmid}/unicode"
            logger.info(f"Falling back to fetch PubMed abstract BioC JSON: {bioc_url}")
            try:
                resp = self._get_with_retry(bioc_url)
                if "[Error]" in resp.text and "No result" in resp.text:
                    logger.warning(f"PubMed abstract BioC JSON not found for {pmid}.")
                else:
                    bioc_data = resp.json()
            except Exception as e:
                logger.error(f"Failed to fetch PubMed abstract BioC JSON for {pmid}: {e}")
                last_error = e

        if not bioc_data:
            raise ValueError(
                f"Failed to retrieve BioC JSON for DOI {doi} (PMCID: {pmcid}, PMID: {pmid}). "
                f"Last error: {last_error}"
            )

        # 3. Save to cache
        if self.cache_dir and bioc_data:
            cache_file = self._get_cache_path(doi)
            try:
                os.makedirs(cache_file.parent, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(bioc_data, f, indent=2)
                logger.debug(f"Saved PMC/PubMed BioC JSON to cache: {doi}")
            except Exception as e:
                logger.error(f"Failed to save PMC/PubMed cache file {cache_file}: {e}")
                
        return bioc_data
