import pytest
import responses
import requests
from unittest.mock import MagicMock, patch
from asma.providers.resolver_crossref import CrossrefResolver
from asma.providers.fetcher_pmc import PmcFetcher
from asma.providers.llm_lmstudio import LMStudioProvider

@responses.activate
def test_crossref_resolver_success():
    doi = "10.1000/xyz123"
    expected_response = {"title": "Test Paper", "DOI": doi}
    
    # Mocking the HTTP request that crossrefapi's Works().doi(doi) will make internally.
    # Crossref API url format: https://api.crossref.org/works/{doi}
    responses.add(
        responses.GET,
        f"https://api.crossref.org/works/{doi}",
        json={"message": expected_response}, # Crossref wraps the body in a "message" envelope usually, but crossrefapi abstracts this. Let's patch Works directly to be safe and clean!
        status=200
    )
    
    resolver = CrossrefResolver()
    with patch.object(resolver.works, 'doi', return_value=expected_response):
        res = resolver.resolve_doi(doi)
        assert res == expected_response

@responses.activate
def test_crossref_resolver_failure():
    doi = "invalid/doi"
    resolver = CrossrefResolver()
    with patch.object(resolver.works, 'doi', return_value=None):
        res = resolver.resolve_doi(doi)
        assert res is None

@responses.activate
def test_pmc_fetcher_success():
    doi = "10.1000/xyz123"
    fetcher = PmcFetcher(email="test@example.com", tool="test_tool")
    
    # 1. Mock the ID Converter API response
    id_conv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={doi}&format=json&tool=test_tool&email=test@example.com"
    id_conv_data = {
        "records": [
            {
                "pmcid": "PMC12345",
                "pmid": "98765"
            }
        ]
    }
    responses.add(responses.GET, id_conv_url, json=id_conv_data, status=200)
    
    # 2. Mock the BioC fetch response
    bioc_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC12345/unicode"
    bioc_data = [{"documents": [{"id": "PMC12345", "passages": []}]}]
    responses.add(responses.GET, bioc_url, json=bioc_data, status=200)
    
    res = fetcher.fetch_by_doi(doi)
    assert res == bioc_data

def test_lmstudio_provider_lazy_loading():
    provider = LMStudioProvider(model_name="test_model")
    assert provider._model is None
    
    with patch("lmstudio.Client") as mock_client_class, patch("lmstudio.set_sync_api_timeout") as mock_set_timeout:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_model = MagicMock()
        mock_client.llm.model.return_value = mock_model
        
        model = provider.model
        assert model == mock_model
        mock_client_class.assert_called_once()
        mock_client.llm.model.assert_called_once_with("test_model")
        mock_set_timeout.assert_called_once_with(3600.0)

def test_lmstudio_provider_context_safety_checks():
    provider = LMStudioProvider(model_name="test_model")
    mock_model = MagicMock()
    mock_model.apply_prompt_template.return_value = "prompt"
    mock_model.tokenize.return_value = [1] * 2000 # 2000 tokens
    mock_model.get_context_length.return_value = 1000 # 1000 limit
    
    provider._model = mock_model
    
    # Should raise ValueError because 2000 > 1000
    with pytest.raises(ValueError, match="exceeds the model's context window limit"):
        provider.respond("test prompt")
        
    # Should pass when bypass option is given
    mock_model.respond.return_value = MagicMock(content="Success response")
    res = provider.respond("test prompt", ignore_context_limit=True)
    assert res == "Success response"

@responses.activate
def test_pmc_fetcher_fallback_to_pubmed():
    doi = "10.1000/xyz123"
    fetcher = PmcFetcher(email="test@example.com", tool="test_tool")
    
    # 1. Mock the ID Converter API response
    id_conv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={doi}&format=json&tool=test_tool&email=test@example.com"
    id_conv_data = {
        "records": [
            {
                "pmcid": "PMC12345",
                "pmid": "98765"
            }
        ]
    }
    responses.add(responses.GET, id_conv_url, json=id_conv_data, status=200)
    
    # 2. Mock the BioC fetch response from PMC returning the "No result can be found" error page
    bioc_pmc_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC12345/unicode"
    responses.add(responses.GET, bioc_pmc_url, body="[Error]: No result can be found.", status=200)
    
    # 3. Mock the BioC fallback response from PubMed Central abstract endpoint
    bioc_pubmed_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/98765/unicode"
    bioc_data = [{"documents": [{"id": "98765", "passages": [{"text": "Abstract text only"}]}]}]
    responses.add(responses.GET, bioc_pubmed_url, json=bioc_data, status=200)
    
    res = fetcher.fetch_by_doi(doi)
    assert res == bioc_data


def test_crossref_resolver_retry():
    doi = "10.1000/retry"
    resolver = CrossrefResolver()
    
    mock_doi = MagicMock()
    expected_response = {"title": "Test Paper", "DOI": doi}
    
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise requests.exceptions.RequestException("Timeout")
        return expected_response
        
    with patch.object(resolver.works, 'doi', side_effect=side_effect):
        res = resolver.resolve_doi(doi)
        assert res == expected_response
        assert call_count == 2


