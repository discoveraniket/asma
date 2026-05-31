from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class MetadataResolver(ABC):
    @abstractmethod
    def resolve_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Validate a DOI and return a dictionary of its metadata.
        Returns None if validation fails.
        """
        pass
