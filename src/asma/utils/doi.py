import logging
from typing import Optional
from asma.interfaces.resolver import MetadataResolver

logger = logging.getLogger(__name__)

def validate_doi(
    doi: str, 
    method: str = "crossref", 
    resolver: Optional[MetadataResolver] = None
) -> bool:
    """
    Validates a DOI using the specified validation method or a custom resolver.
    
    Args:
        doi: The DOI string to validate.
        method: The registry method to use. Supported: 'crossref'.
        resolver: An optional custom MetadataResolver instance to override default resolution.
        
    Returns:
        True if the DOI resolves successfully, False otherwise.
    """
    if resolver is not None:
        logger.info(f"Validating DOI '{doi}' using custom resolver class.")
        return resolver.resolve_doi(doi) is not None

    if method.lower() == "crossref":
        from asma.providers.resolver_crossref import CrossrefResolver
        active_resolver = CrossrefResolver()
        result = active_resolver.resolve_doi(doi)
        return result is not None
    else:
        logger.error(f"Unsupported validation method specified: {method}")
        raise ValueError(f"Unsupported validation method: '{method}'. Currently supported: 'crossref'")
