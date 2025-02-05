"""
This test script will be used to test the API endpoints for the SciNoBo Metadata Mapper
"""
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") 
# ------------------------------------------------------------ #
from fastapi.testclient import TestClient
from metadata_mapper.server.api import app

client = TestClient(app)

def test_echo():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "cosine",
        "k": 10,
        "rerank": True
    }
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    assert response.json() == payload

def test_echo_empty_approach():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra"
    }
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    assert response.json()["approach"] == "cosine"
    assert response.json()["k"] == 10
    assert response.json()["rerank"] is True

def test_search_fos_taxonomy():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "elastic",
        "k": 5,
        "rerank": False
    }
    response = client.post("/search/fos_taxonomy", json=payload)
    assert response.status_code == 200
    assert "retrieved_results" in response.json()
    assert isinstance(response.json()["retrieved_results"], list)

def test_search_fos_taxonomy_no_text():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "",
        "approach": "cosine",
        "k": 5
    }
    response = client.post("/search/fos_taxonomy", json=payload)
    assert response.status_code == 200
    assert response.json()["text"] == "Error reason: no text provided"

def test_search_publication_venues():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum computing conference",
        "approach": "elastic",
        "k": 3
    }
    response = client.post("/search/publication_venues", json=payload)
    assert response.status_code == 200
    assert "retrieved_results" in response.json()
    assert isinstance(response.json()["retrieved_results"], list)

def test_search_affiliations():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "University of Quantum Research",
        "approach": "cosine",
        "k": 7
    }
    response = client.post("/search/affiliations", json=payload)
    assert response.status_code == 200
    assert "retrieved_results" in response.json()
    assert isinstance(response.json()["retrieved_results"], list)

def test_invalid_approach():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "invalid_approach",
        "k": 10
    }
    response = client.post("/search/fos_taxonomy", json=payload)
    assert response.status_code == 422

def test_missing_required_field():
    payload = {
        "text": "quantum algebra",
        "approach": "cosine",
        "k": 10
    }
    response = client.post("/search/fos_taxonomy", json=payload)
    assert response.status_code == 422
