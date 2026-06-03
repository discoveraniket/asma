import pytest
import responses
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from asma.core.ingester import DocumentIngester
from asma.providers.resolver_crossref import CrossrefResolver
from asma.providers.fetcher_pmc import PmcFetcher

@pytest.fixture
def temp_cache_dir():
    # Setup temporary directory for cache testing
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)

@responses.activate
def test_document_ingester_success(temp_cache_dir):
    doi = "10.1000/xyz123"
    
    # 1. Mock Resolver Metadata
    doi_meta = {
        "title": ["Test Document Ingest Title"],
        "author": [{"family": "Doe", "given": "John"}, {"family": "Smith", "given": "Alice"}],
        "container-title": ["Journal of Testing"],
        "published-print": {"date-parts": [[2026]]}
    }
    
    # 2. Mock PMC fetch response
    id_conv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={doi}&format=json&tool=asma_extractor&email=test%40domain.com"
    id_conv_data = {"records": [{"pmcid": "PMC99999", "pmid": "88888"}]}
    responses.add(responses.GET, id_conv_url, json=id_conv_data, status=200)
    
    bioc_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC99999/unicode"
    bioc_data = [{
        "source": "PMC",
        "date": "2026",
        "key": "pmc.key",
        "infons": {},
        "documents": [{
            "id": "PMC99999",
            "infons": {},
            "relations": [],
            "annotations": [],
            "passages": [
                {
                    "infons": {"type": "front", "article-id_doi": doi},
                    "text": "Document Title",
                    "offset": 0,
                    "sentences": [],
                    "annotations": [],
                    "relations": []
                },
                {
                    "infons": {"type": "paragraph"},
                    "text": "This is a body paragraph text.",
                    "offset": 100,
                    "sentences": [],
                    "annotations": [],
                    "relations": []
                }
            ]
        }]
    }]
    responses.add(responses.GET, bioc_url, json=bioc_data, status=200)

    ingester = DocumentIngester(ncbi_email="test@domain.com", cache_dir=temp_cache_dir)
    
    with patch.object(ingester.resolver.works, 'doi', return_value=doi_meta):
        result = ingester.ingest_doi(doi)
        
        assert result["doi"] == doi
        assert result["title"] == "Test Document Ingest Title"
        assert result["authors"] == "Doe J. & Smith A. (2026)"
        assert result["journal"] == "Journal of Testing"
        assert "This is a body paragraph text." in result["cleanContent"]

def test_resolver_caching_behavior(temp_cache_dir):
    doi = "10.2000/cache-test"
    resolver = CrossrefResolver(cache_dir=temp_cache_dir)
    
    doi_meta = {"title": ["Cached Title"], "author": []}
    
    # Verify cache starts empty
    cache_path = resolver._get_cache_path(doi)
    assert not cache_path.exists()
    
    # Resolve first time - mocks API call
    with patch.object(resolver.works, 'doi', return_value=doi_meta) as mock_doi:
        res1 = resolver.resolve_doi(doi)
        assert res1 == doi_meta
        mock_doi.assert_called_once_with(doi)
        
    # Cache file should be created now
    assert cache_path.exists()
    
    # Resolve second time - should read from cache without calling the works.doi mock!
    with patch.object(resolver.works, 'doi') as mock_doi_2:
        res2 = resolver.resolve_doi(doi)
        assert res2 == doi_meta
        mock_doi_2.assert_not_called()
