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
    index_name: str = "fos_taxonomy_01_embed"
    fos_taxonomy_version: str = "v0.1.1"