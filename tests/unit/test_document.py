import sys
import pytest
from unittest.mock import MagicMock, patch
from amas.utils.document import extract_doi_from_pdf

def test_extract_doi_from_pdf_success():
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Random text on page 1."
    
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Some other text. DOI: 10.1000/xyz123-abc End of page."
    
    mock_pdf = MagicMock()
    mock_pdf.__len__.return_value = 2
    mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]
    
    with patch("fitz.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        doi = extract_doi_from_pdf("dummy.pdf", max_pages=3)
        assert doi == "10.1000/xyz123-abc"
        
        called_path = mock_open.call_args[0][0]
        assert str(called_path).endswith("dummy.pdf")

def test_extract_doi_from_pdf_custom_pattern_and_pages():
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 doesn't have it."
    
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 text."
    
    mock_page3 = MagicMock()
    mock_page3.get_text.return_value = "CUSTOM-ID-12345"
    
    mock_pdf = MagicMock()
    mock_pdf.__len__.return_value = 3
    mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2, mock_page3]
    
    with patch("fitz.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        # Test custom pattern and custom page limit
        custom_id = extract_doi_from_pdf(
            "dummy.pdf",
            max_pages=3,
            doi_pattern=r'CUSTOM-ID-\d+'
        )
        assert custom_id == "CUSTOM-ID-12345"

def test_extract_doi_from_pdf_no_doi():
    mock_page = MagicMock()
    mock_page.get_text.return_value = "No DOI pattern anywhere."
    
    mock_pdf = MagicMock()
    mock_pdf.__len__.return_value = 1
    mock_pdf.__getitem__.side_effect = [mock_page]
    
    with patch("fitz.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_pdf
        with pytest.raises(ValueError, match="No DOI matching pattern found"):
            extract_doi_from_pdf("dummy.pdf")

def test_extract_doi_from_pdf_import_error():
    # Force ImportError on import of 'fitz'
    with patch.dict(sys.modules, {'fitz': None}):
        with pytest.raises(ImportError, match="The pymupdf package is required"):
            extract_doi_from_pdf("dummy.pdf")
