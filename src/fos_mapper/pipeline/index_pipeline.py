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
    "--cache_folder", "/storage1/sotkot/llm_models",
    "--delete_index", "False"
]
"""

import os
import argparse
import json
import importlib.resources
import distutils.util

from tqdm import tqdm
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from InstructorEmbedding import INSTRUCTOR


##############################
# DATA_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0]), "data") # TODO uncomment when converted to a package
DATA_PATH = os.path.join("/storage2/sotkot/scinobo-taxonomy-mapper/src/fos_mapper/data")
##############################


class Indexer:
    def __init__(self, index_name, ips, mapping=None, batch_size=20, delete_index=False, mapping_type=None):
        """ ELASTIC CONNECTION """
        self.es = Elasticsearch(ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
        self.index_name = index_name
        self.mapping = mapping
        self.mapping_type = mapping_type
        self.actions = []
        self.batch_size = batch_size
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
                    if 'ConnectionTimeout' in str(e):
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
    parser.add_argument("--cache_folder", type=str, help="Cache folder to store the embeddings.")
    parser.add_argument("--delete_index", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="Delete the index if it exists.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size to index the data.")
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
            
    
def main():
    ################
    args = parse_args()
    index_name = args.index_name
    embedding_model = args.embedding_model
    batch_size = args.batch_size
    cache_folder = args.cache_folder
    delete_index = args.delete_index
    es_host = args.es_host
    es_port = args.es_port
    ################
    # load data
    fos_taxonomy = load_json(os.path.join(DATA_PATH, "fos_taxonomy_0.1.0.json"))
    fos_taxonomy_instruction = load_json(os.path.join(DATA_PATH, "fos_taxonomy_instruction_0.1.0.json"))
    fos_taxonomy_mapping = load_json(os.path.join(DATA_PATH, "fos_taxonomy_mapping_0.1.0.json"))
    ################
    indexer = Indexer(
        index_name=index_name,
        ips=[f"http://{es_host}:{es_port}"],
        mapping=fos_taxonomy_mapping,
        delete_index=delete_index
    )
    # split the taxonomy into batches
    batches = [fos_taxonomy[i:i + batch_size] for i in range(0, len(fos_taxonomy), batch_size)]
    # init the model
    model = INSTRUCTOR(embedding_model, cache_folder=cache_folder, device="cpu")
    for batch in tqdm(batches, desc="Parsing the batches of the taxonomy"):
        # pack the data for embedding
        data = [
            [
                fos_taxonomy_instruction["embed_instruction"],
                f"{b['level_2']}/{b['level_3']}/{b['level_4']}/{b['level_6'].replace(' ---- ', ', ')}"    
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
                "level_6": b["level_6"],
                "fos_vector": e.tolist()
            } for b, e in zip(batch, data_embeddings)
        ]
        for doc in batch_to_index:
            indexer.process_one_dato(doc, op_type="index")
    indexer.upload_to_elk(finished=True)
    

if __name__ == "__main__":
    main()