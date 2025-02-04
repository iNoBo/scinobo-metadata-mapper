from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import traceback

from typing import Union, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from server.logging_setup import setup_root_logger
from pipeline.retriever import Retriever
from utils.data_handling import load_json, load_data_and_instructions
from enum import Enum
from server.config_settings import Settings
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache
def get_settings():
    return Settings()

##############################################################################################################
load_dotenv()
setup_root_logger()
settings = get_settings()
LOGGER = logging.getLogger(__name__)
LOGGER.info("Metadata Mapper API initialized")
MAPPER_HOST= os.environ["MAPPER_HOST"]
MAPPER_PORT=int (os.environ["MAPPER_PORT"])
DATA_PATH =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
MODEL_ARTEFACTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_artefacts")
##############################################################################################################

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# the FastAPI app
app = FastAPI()

# handle CORS -- at a later stage we can restrict the origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ApproachName(Enum):
    elastic = "elastic"
    cosine = "cosine"
    hybrid = "hybrid"

class MapperSearchRequest(BaseModel):
    id: str
    text: Optional[str] = ""
    k: int = 10
    approach: ApproachName = ApproachName.cosine
    rerank: bool = True

class MappedFoSResults(BaseModel):
    fos_label: Optional[str] = None
    level: Optional[str] = None
    score: float = 0.0
    reranker_score: Optional[float] = None

class MappedVenueResults(BaseModel):
    venue_name: Optional[str] = None
    full_name: Optional[str] = None
    venue_id: Optional[str] = None
    url: Optional[str] = None
    score: float = 0.0
    reranker_score: Optional[float] = None

class MappedAffiliationsResults(BaseModel):
    affiliation: Optional[str] = None
    aff_type: Optional[str] = None
    full_name: Optional[str] = None
    uncleaned_name: Optional[str] = None
    score: float = 0.0
    reranker_score: Optional[float] = None

class MapperInferRequestResponse(BaseModel):
    id: str
    text: Optional[str] = ""
    retrieved_results: List[Union[MappedFoSResults, MappedVenueResults, MappedAffiliationsResults]] = []

# Initialize Retriever
retriever = Retriever(
    ips=f"http://{settings.es_host}:{settings.es_port}",
    embedding_model=settings.embedding_model,
    reranker_model=settings.reranker_model,
    device=settings.device,
    cache_dir=MODEL_ARTEFACTS_PATH,
    log_path=LOGGING_PATH,
    es_passwd=settings.es_passwd,
)

def perform_search(request_data: MapperSearchRequest, retriever: Retriever, index: str, version: str, data_type:str):
    """
    Perform dense or lexical search in Elasticsearch index using the Retriever.
    """

    try:
        # Pass index name
        retriever.index = index
        data, instructions, mapping_schema = load_data_and_instructions(file_prefix=data_type, version=version)
        # Instruction for generating query embedding 
        # NOTE: This can be omitted in the future if the instructions prompt remains consistent across all data types."
        retriever.instruction = instructions["query_instruction"]

        LOGGER.info(f"Request data: {request_data}")

        # No text error handling
        if not request_data.text:
            error_response = {
                "id": request_data.id,
                "text": "Error reason: no text provided",
                "retrieved_results": [],
            }
            LOGGER.info(error_response)
            return MapperInferRequestResponse(**error_response)

        # Retrieve
        results = retriever.search_elastic(
            query=request_data.text.lower(),
            how_many=request_data.k,
            approach=request_data.approach.value,
        )

        # Rerank (if requested)
        if request_data.rerank:
            results["hits"] = retriever.rerank_hits(query=request_data.text.lower(), hits=results["hits"], how_many=request_data.k)
        
        # Format results
        if data_type == "fos_taxonomy":
            retrieved_results = [
                MappedFoSResults(
                    fos_label=hit.get("_source", {}).get("fos_label"),
                    level=hit.get("_source", {}).get("level"),
                    score=hit.get("_score", 0.0),
                    reranker_score=hit.get("_reranker_score", None),
                )
                for hit in results.get("hits", [])
            ]
        elif data_type== "publication_venues":
            retrieved_results = [
                MappedVenueResults(
                    venue_name=hit.get("_source", {}).get("venue_name"),
                    full_name=hit.get("_source", {}).get("full_name"),
                    venue_id = hit.get("_source", {}).get("venue_id"),
                    url = hit.get("_source", {}).get ("url"),
                    score=hit.get("_score", 0.0),
                    reranker_score=hit.get("_reranker_score", 0.0),
                )
                for hit in results.get("hits", [])
            ]
        elif data_type == "affiliations":
            retrieved_results = [
                MappedAffiliationsResults(
                    affiliation=hit.get("_source", {}).get("affiliation"),
                    aff_type = hit.get("_source", {}).get("type"),
                    full_name = hit.get("_source", {}).get("full_name"),
                    uncleaned_name=hit.get("_source", {}).get("uncleaned_name"),
                    score=hit.get("_score", 0.0),
                    reranker_score=hit.get("_reranker_score", 0.0),
                )
                for hit in results.get("hits", [])
            ]
        else:
            raise ValueError(f"Unsupported data_type: {request_data.data_type}")
        
        # Construct final response
        response = {
            "id": request_data.id,
            "text": request_data.text,
            "retrieved_results": retrieved_results,
        }
        LOGGER.info(f"Response data: {response}")
        return MapperInferRequestResponse(**response)

    except Exception as e:
        LOGGER.error(f"Error during search: {e}")
        raise HTTPException(status_code=400,detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})

# create a middleware that logs the requests -- this function logs everything. It might not be needed.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    LOGGER.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

# create an endpoint which receives a request and just returns the request data
@app.post("/echo", response_model=MapperSearchRequest)
def echo(request_data: MapperSearchRequest):
    LOGGER.info(f"Echo request data: {request_data}")
    return request_data

@app.post("/search/fos_taxonomy", response_model=MapperInferRequestResponse)
def search_fos_taxonomy(
    request_data: MapperSearchRequest,
    index="fos_taxonomy_labels_02_embed",
    version="v0.1.2"
    ):
    
    "Infer the field of science mapping of a query. The response data contains the top k most similar FoS labels (across all levels)."
    return perform_search(request_data, retriever, index, version, data_type="fos_taxonomy")

@app.post("/search/publication_venues", response_model=MapperInferRequestResponse)
def search_publication_venues(
    request_data: MapperSearchRequest, index="publication_venues_embed_v1.0", version="v1.0"
    ):

    "Infer the publication venue (e.g., journal) mapping of a query. The response data contains the top k most similar venue names and their corresponding full names."
    return perform_search(request_data, retriever, index, version, data_type = "publication_venues")

@app.post("/search/affiliations", response_model=MapperInferRequestResponse)
def search_affiliations(
    request_data: MapperSearchRequest, index="affiliation_names_embed_v2.0", version="v2.0"
    ):
    
    "Infer the affiliation (e.g., University) mapping of a query. The response data contains the top k most similar affiliation names."
    return perform_search(request_data, retriever, index, version, data_type= "affiliations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=MAPPER_HOST, port=MAPPER_PORT)