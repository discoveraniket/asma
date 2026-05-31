import logging
from typing import Dict, Any, Optional
from crossref.restful import Works
from amas.interfaces.resolver import MetadataResolver

logger = logging.getLogger(__name__)

class CrossrefResolver(MetadataResolver):
    """
    Validates DOIs and fetches their metadata using the Crossref Works API.
    """
    def __init__(self):
        self.works = Works()

    def resolve_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Validating DOI on Crossref: {doi}")
        try:
            result = self.works.doi(doi)
            if result is None:
                logger.warning(f"Crossref could not find DOI: {doi}")
                return None
            logger.info("DOI validation succeeded.")
            return result
        except Exception as e:
            logger.error(f"Error validating DOI {doi} on Crossref: {e}")
            return None
