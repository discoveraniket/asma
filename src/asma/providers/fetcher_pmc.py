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

        if pmcid:
            target_id = pmcid
            api_endpoint = "pmcoa.cgi"
            logger.info(f"Resolved target ID to PMCID: {target_id}")
        elif pmid:
            target_id = pmid
            api_endpoint = "pubmed.cgi"
            logger.info(f"Resolved target ID to PMID: {target_id}")
        else:
            raise ValueError(f"DOI {doi} conversion failed to yield a PMCID or PMID.")

        bioc_url = f"{self.bionlp_base_url}/{api_endpoint}/BioC_json/{target_id}/unicode"
        logger.info(f"Fetching BioC JSON from {bioc_url}...")
        
        try:
            resp = self._get_with_retry(bioc_url)
            bioc_data = resp.json()
            
            # 3. Save to cache
            if self.cache_dir and bioc_data:
                cache_file = self._get_cache_path(doi)
                try:
                    os.makedirs(cache_file.parent, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(bioc_data, f, indent=2)
                    logger.debug(f"Saved PMC BioC JSON to cache: {doi}")
                except Exception as e:
                    logger.error(f"Failed to save PMC cache file {cache_file}: {e}")
                    
            return bioc_data
        except Exception as e:
            logger.error(f"Failed to fetch BioC JSON from {bioc_url}: {e}")
            raise ValueError(f"Failed to retrieve BioC JSON for target ID {target_id}: {e}") from e
