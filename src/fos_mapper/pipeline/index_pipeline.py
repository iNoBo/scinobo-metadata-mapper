""" 

- This script contains the functions and class to index data to a provided elasticsearch installation.
- This script is provided an embedding model (more specifically instructor-xl or any other version)
to embed the data before indexing it to elasticsearch.
- This script is also provided a data source to index the data from.
- This script is also provided a target index name to index the data to.
- This script is also provided a target index mapping to index the data with.
- This script is also provided an instruction for the instructor-xl model to embed the data with.

NOTE For now we do not pass any credentials. However, in the future this must change.

args = [
    "--index_name", "fos_taxonomy_01_embed",
    "--embedding_model", "instructor-xl",
    "--es_host", "localhost",
    "--es_port", "9200" ,
    "--batch_size", "16",
    "--device", "cuda",
    "--cache_folder", "/storage1/sotkot/llm_models",
    "--delete_index", "False",
    "--fos_taxonomy_version", "v0.1.1"
]
"""

import os
import argparse
import json
import distutils.util
import importlib.resources

from tqdm import tqdm
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

##############################
#DATA_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "data")
#MODEL_ARTEFACTS = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "model_artefacts")
DATA_PATH = "/workspaces/scinobo-taxonomy-mapper/src/fos_mapper/data"
MODEL_ARTEFACTS = "/workspaces/scinobo-taxonomy-mapper/src/fos_mapper/model_artefacts"
##############################


class Indexer:
    def __init__(self, index_name, es_passwd, ips, mapping=None, batch_size=1000, delete_index=False, mapping_type=None, ca_certs_path=None):
        """ ELASTIC CONNECTION """
        if ca_certs_path is None:
            self.es = Elasticsearch(
                ips, 
                max_retries=10, 
                retry_on_timeout=True,
                request_timeout=150,
                basic_auth=('elastic', es_passwd)
            )
        else:
            self.es = Elasticsearch(
                ips, 
                ca_certs=ca_certs_path, 
                max_retries=10, 
                retry_on_timeout=True, 
                request_timeout=150,
                basic_auth=('elastic', es_passwd)
            )
        self.index_name = index_name
        self.mapping = mapping
        self.mapping_type = mapping_type
        self.actions = []
        self.batch_size = batch_size # this is for indexing the data
        # check if index exists
        if not self.check_if_index_exists():
            self.create_index()
        else:
            print(f'Index {self.index_name} already exists')
            if delete_index:
                self.delete_index()
                self.create_index()
            else:
                print(f'Index {self.index_name} already exists, force delete it first if you want, by parsing argument --delete_index True in indexer class')
    
    def delete_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        
    def check_if_index_exists(self):
        return self.es.indices.exists(index=self.index_name)
    
    def create_index(self):
        if self.mapping is None:
            raise Exception('No mapping provided')
        self.es.indices.create(index=self.index_name, body=self.mapping)
        
    def create_an_action(self, dato, op_type, the_id=None):
        if the_id is None:
            pass
        else:
            dato['_id'] = the_id
        ################
        dato['_op_type'] = op_type
        dato['_index'] = self.index_name
        return dato
    
    def upload_to_elk(self, finished=False):
        if (len(self.actions) >= self.batch_size) or (len(self.actions) > 0 and finished):
            flag = True
            while flag:
                try:
                    _ = bulk(self.es, iter(self.actions))
                    # pprint(result)
                    flag = False
                except Exception as e:
                    print(e)
                    if 'ConnectionTimeout' in str(e) or 'Connection timed out' in str(e):
                        print('Retrying')
                    else:
                        flag = False
            self.actions = []
    
    def process_folder(self, data):
        for dato in data:
            self.process_one_dato(dato)
            
    def process_one_dato(self, doc, op_type, the_id=None):
        self.actions.append(self.create_an_action(
            dato=doc, 
            op_type=op_type,
            the_id=the_id
        ))
        self.upload_to_elk(finished=False)
        

def parse_args():
    parser = argparse.ArgumentParser(description="Index data to elasticsearch.")
    parser.add_argument("--index_name", type=str, help="Name of the index to index the data to.")
    parser.add_argument("--embedding_model", type=str, help="Name of the embedding model.")
    parser.add_argument("--es_host", type=str, help="Elasticsearch host.")
    parser.add_argument("--es_port", type=str, help="Elasticsearch port.")
    parser.add_argument("--es_passwd", type=str, help="Elasticsearch password.")
    parser.add_argument("--ca_certs", type=str, help="Path to the ca certs.", default=None)
    parser.add_argument("--device", type=str, help="The type of the device")
    parser.add_argument("--cache_folder", type=str, help="Cache folder to store the embeddings.")
    parser.add_argument("--delete_index", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="Delete the index if it exists.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to embed the data.")
    parser.add_argument("--fos_taxonomy_version", type=str, help="Version of the fos taxonomy.")
    parser.add_argument("--index_only_fos_labels", action = "store_true", help="Whether to index solely the FoS names along with their corresponding levels individually, without including the entire taxonomy paths.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    

def compute_embeddings(model, documents):
    """Compute dense embeddings.

    Parameters
    ----------
    model : SentenceTransformer
        sentence transformer model instance
    documents : list of str
        list of processed/clean documents
        it is a list of a list. Each inner list contains the 
        instruction and the document to embed.
    Returns
    ------
    list of numpy.ndarray
        a list of document vectors
    """
    embeddings = model.encode(documents, show_progress_bar=True)
    return embeddings

def split_into_batches(data:list, batch_size:int):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
def main():
    ################
    args = parse_args()
    index_name = args.index_name
    embedding_model = args.embedding_model
    batch_size = args.batch_size
    device = args.device
    cache_folder = args.cache_folder
    delete_index = args.delete_index
    es_host = args.es_host
    es_port = args.es_port
    es_passwd = args.es_passwd
    ca_certs_path = args.ca_certs
    fos_taxonomy_version = args.fos_taxonomy_version
    index_only_fos_labels = args.index_only_fos_labels
    ################
    # load data
    fos_taxonomy_in = load_json(os.path.join(DATA_PATH, f"fos_taxonomy_{fos_taxonomy_version}.json"))
    fos_taxonomy_instruction = load_json(os.path.join(DATA_PATH, f"fos_taxonomy_instruction_{fos_taxonomy_version}.json"))
    mapping_schema_file = (
        os.path.join(DATA_PATH, f"fos_labels_mapping_{fos_taxonomy_version}.json")
        if index_only_fos_labels
        else os.path.join(DATA_PATH, f"fos_taxonomy_mapping_{fos_taxonomy_version}.json")
    )
    mapping_schema = load_json(mapping_schema_file)
    ################
    # if the fos_taxonomy is a dict, convert it to a list by flattening the values
    if isinstance(fos_taxonomy_in, dict):
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
            } for value in fos_taxonomy_in.values() for v in value
        ]
    else:
        # lower case all the values
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
            } for b in fos_taxonomy_in
        ]
    ################
    indexer = Indexer(
        index_name=index_name,
        es_passwd=es_passwd,
        ca_certs_path=ca_certs_path,
        ips=[f"https://{es_host}:{es_port}"] if ca_certs_path is not None else [f"http://{es_host}:{es_port}"],
        mapping=mapping_schema,
        delete_index=delete_index
    )
    # init the model
    model = SentenceTransformer(
        embedding_model = embedding_model, 
        cache_folder=cache_folder if cache_folder is not None else MODEL_ARTEFACTS, 
        device=device,
        trust_remote_code=True
    )

    if index_only_fos_labels:
        # Create a set of FoS label-level pairs for indexing
        fos_labels_set = set()
        for entry in fos_taxonomy:
            for level in ["level_1", "level_2", "level_3", "level_4", "level_5_name"]:
                label = entry.get(level)
                if label != "n/a":
                    fos_labels_set.add((label, level))

        # Index fos labels seperately
        batches = split_into_batches(list(fos_labels_set), batch_size)
        for batch in tqdm(batches, desc="Indexing FoS labels"):
            data = [ fos_taxonomy_instruction["embed_instruction"] + b[0] for b in batch]
            data_embeddings = compute_embeddings (model, data)
            batch_to_index = [{"fos_label": b[0], "level": b[1], "fos_vector": e.tolist()} for b, e in zip(batch, data_embeddings)]
            
            for doc in batch_to_index:
                indexer.process_one_dato(doc, op_type="index")
    else:
        # Index full fos taxonomy paths
        batches = split_into_batches(fos_taxonomy, batch_size)
        for batch in tqdm(batches, desc="Parsing the batches of the taxonomy"):
            # pack the data for embedding
            data = [
                [
                    fos_taxonomy_instruction["embed_instruction"] + 
                    f"{b['level_2']}/{b['level_3']}/{b['level_4']}/{b['level_6'].replace(' ---- ', ', ')}"
                    if b['level_6'] != "n/a" 
                    else f"{b['level_2']}/{b['level_3']}/{b['level_4']}" 
                    if b['level_4'] != "n/a" 
                    else f"{b['level_2']}/{b['level_3']}" if b['level_3'] != "n/a" else f"{b['level_2']}"
                ] for b in batch
            ]
            data_embeddings = compute_embeddings(model, data)
            batch_to_index = [
                {
                    "level_1": b["level_1"],
                    "level_2": b["level_2"],
                    "level_3": b["level_3"],
                    "level_4": b["level_4"],
                    "level_4_id": b["level_4_id"],
                    "level_5_id": b["level_5_id"],
                    "level_5": b["level_5"],
                    "level_5_name": b["level_5_name"],
                    "level_6": b["level_6"],
                    "fos_vector": e.tolist()
                } for b, e in zip(batch, data_embeddings)
            ]
            for doc in batch_to_index:
                indexer.process_one_dato(doc, op_type="index")
    indexer.upload_to_elk(finished=True)
    

if __name__ == "__main__":
    main()