import re
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

DEFAULT_DOI_PATTERN = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'

def extract_doi_from_pdf(
    pdf_path: Union[str, Path],
    max_pages: int = 2,
    doi_pattern: str = DEFAULT_DOI_PATTERN
) -> str:
    """
    Extracts DOI from the first few pages of a local PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        max_pages: Maximum number of pages from the beginning of the PDF to scan (default is 2).
        doi_pattern: The regex pattern to match the DOI (default is the standard DOI pattern).
        
    Returns:
        The extracted DOI string.
        
    Raises:
        ImportError: If pymupdf is not installed.
        ValueError: If no DOI pattern is found in the specified pages.
    """
    try:
        import fitz  # Lazy Import of PyMuPDF (fitz)
    except ImportError as e:
        logger.error("pymupdf (fitz) is not installed. Install it with: pip install amas[pdf]")
        raise ImportError(
            "The pymupdf package is required for PDF DOI extraction. "
            "Please install it using 'pip install amas[pdf]' or 'pip install pymupdf'."
        ) from e

    path = Path(pdf_path)
    logger.info(f"Opening PDF file for DOI extraction: {path} (scanning first {max_pages} pages)")
    
    with fitz.open(path) as pdf:
        text = ""
        for i in range(min(max_pages, len(pdf))):
            text += pdf[i].get_text() or ""
        
        match = re.search(doi_pattern, text, re.IGNORECASE)
        
        if match:
            doi = match.group(0)
            logger.info(f"Found DOI: {doi}")
            return doi
        else:
            raise ValueError(f"No DOI matching pattern found in the first {max_pages} pages of PDF: {path}")
