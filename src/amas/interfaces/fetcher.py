from abc import ABC, abstractmethod
from typing import Dict, Any

class ArticleFetcher(ABC):
    @abstractmethod
    def fetch_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Resolves the DOI to database identifiers (e.g. PMC/PubMed IDs)
        and fetches the raw document structure (e.g. BioC JSON).
        
        Raises ValueError if resolution or fetching fails.
        """
        pass
