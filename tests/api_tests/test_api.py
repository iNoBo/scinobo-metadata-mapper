""" 

This test script will be used to test the API endpoints for the SciNoBo-FWCI.

"""
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from fastapi.testclient import TestClient
from fos_mapper.server.api import app


client = TestClient(app)

def test_echo():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "knn"
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
    assert response.json()["approach"] == "knn"
    

def test_infer_mapper():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "knn"
    }
    response = client.post("/infer_mapper", json=payload)
    assert response.status_code == 200
    assert all(key in response.json() for key in ["id", "text", "retrieved_results"])
    

def test_infer_mapper_invalid_approach():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "nothing"
    }
    response = client.post("/infer_mapper", json=payload)
    assert response.status_code == 422
    

def test_infer_mapper_empty_approach():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra"
    }
    response = client.post("/infer_mapper", json=payload)
    assert response.status_code == 200
    assert all(key in response.json() for key in ["id", "text", "retrieved_results"])
    

def test_infer_mapper_empty_text():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "approach": "knn"
    }
    response = client.post("/infer_mapper", json=payload)
    assert response.status_code == 200
    assert response.json()['text'] == 'Error reason : no text'
    

def test_infer_mapper_nonsense_text():
    payload = {
        "id": "10.18653/v1/w19-5032",
        "approach": "knn",
        "text": "asdasdasdasd"
    }
    response = client.post("/infer_mapper", json=payload)
    assert response.status_code == 200
    assert all(key in response.json() for key in ["id", "text", "retrieved_results"])