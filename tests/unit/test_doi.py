import pytest
from unittest.mock import MagicMock, patch
from asma.utils.doi import validate_doi

def test_validate_doi_default_success():
    with patch("asma.providers.resolver_crossref.CrossrefResolver.resolve_doi") as mock_resolve:
        mock_resolve.return_value = {"title": "Test Title"}
        
        is_valid = validate_doi("10.1000/xyz")
        assert is_valid is True
        mock_resolve.assert_called_once_with("10.1000/xyz")

def test_validate_doi_default_failure():
    with patch("asma.providers.resolver_crossref.CrossrefResolver.resolve_doi") as mock_resolve:
        mock_resolve.return_value = None
        
        is_valid = validate_doi("10.1000/xyz")
        assert is_valid is False

def test_validate_doi_custom_resolver():
    mock_resolver = MagicMock()
    mock_resolver.resolve_doi.return_value = {"metadata": "data"}
    
    is_valid = validate_doi("10.1000/xyz", resolver=mock_resolver)
    assert is_valid is True
    mock_resolver.resolve_doi.assert_called_once_with("10.1000/xyz")

def test_validate_doi_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported validation method"):
        validate_doi("10.1000/xyz", method="unpaywall")
