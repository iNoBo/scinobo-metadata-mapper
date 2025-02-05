""" 

This script is used for passing env variables to the FastAPI server.
We also overriding them for testing.

"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    es_passwd: str
    ca_certs_path: str | None = None
    device: str = "cuda"
    es_host: str = "localhost"
    es_port: str = "9200"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
