import logging
import hashlib
import json
import os
import backoff
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from crossref.restful import Works
from asma.interfaces.resolver import MetadataResolver

logger = logging.getLogger(__name__)

def format_crossref_authors(authors_list: List[Dict[str, Any]]) -> str:
    """
    Standardizes Crossref author structures into human-readable citation strings.
    """
    formatted = []
    for author in authors_list:
        family = author.get('family', '')
        given = author.get('given', '')
        initial = f" {given[0]}." if given else ""
        formatted.append(f"{family}{initial}")
    if not formatted:
        return "Unknown Authors"
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    else:
        return f"{formatted[0]} et al."

class CrossrefResolver(MetadataResolver):
    """
    Validates DOIs and fetches their metadata using the Crossref Works API.
    Supports optional disk caching.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.works = Works()
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _get_cache_path(self, doi: str) -> Path:
        hashed = hashlib.sha256(doi.strip().lower().encode('utf-8')).hexdigest()
        return self.cache_dir / "doi_validations" / f"{hashed}.json"

    def resolve_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        doi = doi.strip()
        
        # 1. Check cache first
        if self.cache_dir:
            cache_file = self._get_cache_path(doi)
            if cache_file.exists():
                try:
                    logger.info(f"Crossref DOI validation retrieved from cache: {doi}")
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read Crossref cache file {cache_file}: {e}")

        # 2. Query Crossref API
        logger.info(f"Validating DOI on Crossref: {doi}")
        try:
            @backoff.on_exception(
                backoff.expo,
                requests.exceptions.RequestException,
                max_tries=3
            )
            def _query():
                return self.works.doi(doi)

            result = _query()
            if result is None:
                logger.warning(f"Crossref could not find DOI: {doi}")
                return None
                
            # 3. Save to cache
            if self.cache_dir and result:
                cache_file = self._get_cache_path(doi)
                try:
                    os.makedirs(cache_file.parent, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    logger.debug(f"Saved Crossref metadata to cache: {doi}")
                except Exception as e:
                    logger.error(f"Failed to save Crossref cache file {cache_file}: {e}")
                    
            logger.info("DOI validation succeeded.")
            return result
        except Exception as e:
            logger.error(f"Error validating DOI {doi} on Crossref: {e}")
            return None
