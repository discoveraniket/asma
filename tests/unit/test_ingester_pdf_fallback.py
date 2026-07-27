import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from asma.core.ingester import DocumentIngester

def test_ingest_doi_pmc_success():
    ingester = DocumentIngester()
    
    mock_meta = {
        'title': ['Test Paper Title'],
        'author': [{'given': 'Jane', 'family': 'Doe'}],
        'container-title': ['Test Journal'],
        'created': {'date-parts': [[2026]]}
    }
    
    mock_bioc = MagicMock()
    
    with patch.object(ingester.resolver, 'resolve_doi', return_value=mock_meta), \
         patch.object(ingester.fetcher, 'fetch_by_doi', return_value=mock_bioc), \
         patch('asma.core.ingester.parse_bioc_to_human_markdown', return_value='# Test Clean Content'), \
         patch('asma.core.ingester.parse_bioc_to_llm_markdown', return_value='# Test Raw Content'):
        
        result = ingester.ingest_doi("10.1000/182")
        
        assert result['doi'] == "10.1000/182"
        assert result['title'] == "Test Paper Title"
        assert "Doe" in result['authors']
        assert result['cleanContent'] == "# Test Clean Content"
        assert result['rawContent'] == "# Test Raw Content"


def test_ingest_doi_pmc_failure_pdf_fallback(tmp_path):
    ingester = DocumentIngester()
    
    mock_meta = {
        'title': ['Paywalled Paper Title'],
        'author': [{'given': 'John', 'family': 'Smith'}],
        'container-title': ['Paywalled Journal']
    }
    
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
    
    with patch.object(ingester.resolver, 'resolve_doi', return_value=mock_meta), \
         patch.object(ingester.fetcher, 'fetch_by_doi', side_effect=RuntimeError("PMC 404 Not Found")), \
         patch('pymupdf4llm.to_markdown', return_value="# Extracted Local PDF Markdown\nTable data here"):
        
        result = ingester.ingest_doi("10.1000/paywall123", pdf_path=str(dummy_pdf))
        
        assert result['title'] == "Paywalled Paper Title"
        assert result['cleanContent'] == "# Extracted Local PDF Markdown\nTable data here"
        assert result['rawContent'] == "# Extracted Local PDF Markdown\nTable data here"
