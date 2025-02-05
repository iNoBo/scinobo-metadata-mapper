import json
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

##############################################
load_dotenv()
DATA_PATH =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
##############################################

def load_json(path):
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)  
            return [json.loads(line) for line in f if line.strip()]

def load_data_and_instructions(file_prefix, version):
    """Load data, instructions, and mapping schema for indexing data in Elasticsearch."""

    data_dir = os.path.join (DATA_PATH, file_prefix)
    data = load_json(os.path.join(data_dir, f"{file_prefix}_{version}.json"))
    instructions = load_json(os.path.join(data_dir, f"{file_prefix}_instruction_{version}.json"))
    mapping_schema = load_json(os.path.join(data_dir, f"{file_prefix}_mapping_{version}.json"))
    return data, instructions, mapping_schema

def split_into_batches(data, batch_size):
    """Split data into smaller batches."""

    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def compute_embeddings(model, documents):
    """Compute dense embeddings for a list of documents."""

    return model.encode(documents, show_progress_bar=False)
