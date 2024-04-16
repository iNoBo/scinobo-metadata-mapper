""" 

FastAPI for the FoS taxonomy mapper. This docstring will be updated.

"""
from __future__ import annotations

import logging
import traceback
import os
import importlib.resources
import json

from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fos_mapper.server.logging_setup import setup_root_logger
from fos_mapper.pipeline.inference import Retriever
from enum import Enum


# inits and declarations
# --------------------------------------------------------- #
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

DATA_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "data")
MODEL_ARTEFACTS_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "model_artefacts")
LOGGING_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "logs")
# init the logger
setup_root_logger()
LOGGER = logging.getLogger(__name__)
LOGGER.info("FoS Taxonomy Mapper Api initialized")

fos_taxonomy_instruction = load_json(os.path.join(DATA_PATH, "fos_taxonomy_instruction_0.1.0.json"))
# NOTE when made public or shared as a docker image -- this declaration here must change
# and reveice all the variables as arguments or from a yaml file.
retriever = Retriever(
    ips = [
        "http://localhost:9200"
    ],
    index="fos_taxonomy_01_embed",
    embedding_model="hkunlp/instructor-xl",
    device="cpu",
    instruction=fos_taxonomy_instruction['query_instruction'],
    cache_dir=MODEL_ARTEFACTS_PATH,
    log_path=LOGGING_PATH
)


# declare classes for input-output and error responses
class ApproachName(Enum):
    """_summary_
        Describes the approach of performing the inference. 
        The view can be one of the following: "knn", "elastic", "cosine".
    Args:
        Enum (_type_): _description_
    """
    knn = "knn"
    elastic = "elastic"
    cosine = "cosine"
    
    
class MapperInferRequest(BaseModel):
    # based on the request config, the request data should contain the following fields
    id: str
    text: str | None = ""
    approach: ApproachName | None = "knn"
    # add an example for the request data
    model_config = {
        "id": "10.18653/v1/w19-5032",
        "text": "quantum algebra",
        "approach": "knn"
    }
    
class MappedResults(BaseModel):
    # based on the request config, the response data should contain the following fields
    level_1: str
    level_2: str
    level_3: str
    level_4: str
    level_4_id: str
    level_5_id: str
    level_5: str
    level_6: str
    
class MapperInferRequestResponse(BaseModel):
    # based on the request config, the response data should contain the following fields
    id: str
    text: str | None = ""
    retrieved_results: list[Union[MappedResults, None]] = []
    
    
# the FastAPI app
app = FastAPI()


# handle CORS -- at a later stage we can restrict the origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create a middleware that logs the requests -- this function logs everything. It might not be needed.
@app.middleware("http")
async def log_requests(request, call_next):
    LOGGER.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# create an endpoint which receives a request and just returns the request data
@app.post("/echo", response_model=MapperInferRequestResponse)
def echo(request_data: MapperInferRequest):
    LOGGER.info(f"Request data for echo: {request_data}")
    return request_data.model_dump() 


@app.post("/infer_mapper", response_model=MapperInferRequestResponse)
def infer_mapper(request_data: MapperInferRequest):
    # TODO: update the docstring, to be more informative.
    """
    Infer the field of science mapping of a query based on the embeddings provided by Instructor. 

    Args:
        request_data (MapperInferRequests): The request data containing the query.

    Returns:
        MapperInferRequestsResponse: The response data containing the inferred field of studies.
    """
    LOGGER.info(f"Request data: {request_data}") # this is formatted based on the BaseModel classes
    try:
        # process the input data to convert it to json -- since we are here, the format is OK. This is handled by FastAPI
        request_data = request_data.model_dump() 
        # sanity checks
        if ('text' not in request_data) or (request_data['text'] is None) or (request_data['text'] == ''):
            ret = {
                'id': request_data['doc_id'],
                'text': 'Error reason : no text',
                'retrieved_results': []
            }
            LOGGER.info(ret)
            return MapperInferRequestResponse(**ret)
        # infer the field of studies
        res = retriever.search_elastic_dense(
            query=request_data['text'],
            approach=request_data['approach'].value
        )
        LOGGER.info(f"Response data: {request_data}") # this is formatted based on the BaseModel classes
        return request_data
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1997)