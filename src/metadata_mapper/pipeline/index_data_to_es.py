import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from indexer import Indexer
from utils.data_handling import compute_embeddings, split_into_batches, load_data_and_instructions
import argparse
import distutils.util
from tqdm import tqdm
from dotenv import load_dotenv

###################################################
load_dotenv()
ES_PORT= os.environ["ES_PORT"]
ES_HOST= os.environ["ES_HOST"]
ES_PASSWD = os.environ["ES_PASSWD"]
CA_CERTS_PATH = os.environ ["CA_CERTS_PATH"]
MODEL_ARTEFACTS=os.environ["MODEL_ARTEFACTS"]
###################################################

def index_fos_taxonomy_labels (data, indexer, model, instructions):
    """
    Process and index FoS taxonomy data. 
    Index entry fields: 
        {
            "fos_label: str,
            "level: str",
            "vector: ndarray"
        }
    """

    # if the fos_taxonomy is a dict, convert it to a list by flattening the values
    if isinstance(data, dict):
        fos_taxonomy = [
            {
                "level_1": v["level_1"].lower(),
                "level_2": v["level_2"].lower(),
                "level_3": v["level_3"].lower(),
                "level_4": v["level_4"].lower(),
                "level_4_id": v["level_4_id"],
                "level_5_id": v["level_5_id"],
                "level_5": v["level_5"].lower(),
                "level_6": v["level_6"].lower(),
                "level_5_name": v["l5_name"].lower()
            } for value in data.values() for v in value
        ]
    else:
        fos_taxonomy = [
            {
                "level_1": b["level_1"].lower(),
                "level_2": b["level_2"].lower(),
                "level_3": b["level_3"].lower(),
                "level_4": b["level_4"].lower(),
                "level_4_id": b["level_4_id"],
                "level_5_id": b["level_5_id"],
                "level_5": b["level_5"].lower(),
                "level_6": b["level_6"].lower(),
                "level_5_name": b["level_5_name"].lower() if "level_5_name" in b else b['l5_name']
            } for b in data
        ]

    # Convert data into structured index entries
    fos_labels_set = set()
    for entry in fos_taxonomy:
        for level in ["level_1", "level_2", "level_3", "level_4", "level_5_name"]:
            label = entry.get(level)
            if label != "n/a":
                fos_labels_set.add((label, level))

    # Split data into batches for processing
    batches = split_into_batches(list(fos_labels_set), indexer.batch_size)

    for batch in tqdm(batches, desc="Indexing FoS labels"):
        data = [instructions["embed_instruction"] + label for label, _ in batch]
        embeddings = compute_embeddings(model, data)
        # Create final index entries
        batch_to_index = [{"fos_label": b[0], "level": b[1], "vector": e.tolist()} for b, e in zip(batch, embeddings)]
        for item in batch_to_index:
            indexer.process_one_dato(item, op_type="index")
    
    indexer.upload_to_elk(finished=True)

def index_venues (data, indexer, model, instructions):
    """
     Process and store venue names to Elasticsearch
     Index entry fields: 
        {
            "venue_name:str",
            "full_name: str,
            "venue_id": str,
            "type: str",
            "vector: ndarray",
        }
    """

    venue_names = []
    
    # Convert data into structured index entries
    for entry in data: 
        full_name = entry["name"].lower()
        alternate_names = set([alt_name.lower() for alt_name in entry["alternate_names"]])
        venue_names.append ({
                    "venue_name": full_name,
                    "full_name": full_name,
                    "venue_id" : entry["id"],
                    "type": entry["type"],
                    "url": entry ["url"]
                })
        if len (alternate_names) != 0:
            for alt_name in alternate_names:
                
                venue_names.append({
                        "venue_name": alt_name.lower(),
                        "full_name": full_name,
                        "venue_id" : entry["id"],
                        "type": entry["type"],
                        "url": entry ["url"]
                    })
                
    # Split data into batches for processing
    batches = split_into_batches(venue_names, indexer.batch_size)

    for batch in tqdm(batches, desc="Indexing venue names"):
        data = [instructions["embed_instruction"] + item["venue_name"] for item in batch]
        embeddings = compute_embeddings (model, data)
        for item, embedding in zip (batch, embeddings):
            item ["vector"] = embedding.tolist ()
            indexer.process_one_dato(item, op_type="index")

    indexer.upload_to_elk(finished=True)


def index_affiliations (data, indexer, model, instructions):
    """
     Process and store affiliation names to Elasticsearch.
     Index entry fields: 
        {
            "affiliation": str",
            "full_name": str,
            "uncleaned_name": str,
            "type": str,
            "vector": ndarray"
        }
    """

    affiliations = []
    # Convert data into structured index entries
    for entry in data: 
        affiliations.append (
                {
                    "affiliation": entry["cleaned"],
                    "full_name": None,
                    "uncleaned_name": entry["original"],
                    "type": "full_name"
                }
                )
        acronyms = set ([acronym for acronym in entry["acronyms"]])
        if len (acronyms) != 0:
            for acronym in acronyms:
                affiliations.append (
                    {
                        "affiliation": acronym.lower(),
                        "full_name": entry["cleaned"],
                        "uncleaned_name": entry["original"],
                        "type": "acronym"

                    }
                )
    # Split data into batches for processing
    batches = split_into_batches(affiliations, indexer.batch_size)

    for batch in tqdm(batches, desc="Indexing affiliation names"):
        data = [instructions["embed_instruction"] + item["affiliation"] for item in batch]
        embeddings = compute_embeddings (model, data)
        for item, embedding in zip (batch, embeddings):
            item ["vector"] = embedding.tolist ()
            indexer.process_one_dato(item, op_type="index")

    indexer.upload_to_elk(finished=True)

def route_and_index(data_file_prefix, data, indexer, model, instructions):
    """Route to the appropriate indexing method based on the data_file_prefix."""

    routing_map = {
        "fos_taxonomy": index_fos_taxonomy_labels,
        "publication_venues": index_venues,
        "affiliations": index_affiliations,
    }
    
    if data_file_prefix in routing_map:
        print ("indexing data...")
        routing_map[data_file_prefix](data, indexer,  model, instructions)
    else:
        raise ValueError(f"Unknown data_file_prefix: {data_file_prefix}")

def main ():
    parser = argparse.ArgumentParser(description="Index data to Elasticsearch.")
    parser.add_argument("--data_file_prefix", type=str, required=True, help="The prefix of the directory and files containing the data to be indexed.", choices = ["fos_taxonomy", "publication_venues", "affiliations"])
    parser.add_argument("--index_name", type=str, required=True, help = "Name of Elasticsearch index.")
    parser.add_argument("--delete_index", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="Delete the index if it exists.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to embed the data.")
    parser.add_argument("--embedding_model", type=str, help="Name of the embedding model to generate dense vectors.", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--device", type=str, help="The type of the device", default="cuda")
    parser.add_argument("--cache_folder", type=str, help="Cache folder to store the embeddings.")
    parser.add_argument("--version", required = True, type=str, help="Version of data file to index.")
    args = parser.parse_args()
    
    # Load data from dir
    data, instructions, mapping_schema = load_data_and_instructions(args.data_file_prefix, args.version)
    # Initialize indexer
    indexer = Indexer(
        index_name=args.index_name,
        ips=f"http://{ES_HOST}:{ES_PORT}",
        es_passwd=ES_PASSWD,
        mapping=mapping_schema,
        batch_size=args.batch_size,
        delete_index=args.delete_index
    )
    # Initialize the embedding model
    model = SentenceTransformer(
        model_name_or_path= args.embedding_model, 
        cache_folder= args.cache_folder if args.cache_folder is not None else MODEL_ARTEFACTS, 
        device=args.device,
        trust_remote_code=True
    )
    # Index data to Elasticsearch
    try:
        route_and_index(data_file_prefix = args.data_file_prefix, data = data, indexer = indexer, model = model,  instructions = instructions)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()