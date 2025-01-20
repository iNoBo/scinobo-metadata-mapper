

# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
import json
import argparse
from dotenv import load_dotenv
import os

from metadata_mapper.pipeline.retriever import Retriever
from pprint import pprint

##############################################
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
ES_PORT= os.environ["ES_PORT"]
ES_HOST= os.environ["ES_HOST"]
ES_PASSWD = os.environ["ES_PASSWD"]
CA_CERTS_PATH = os.environ ["CA_CERTS_PATH"]
MODEL_ARTEFACTS=os.environ["MODEL_ARTEFACTS"]
##############################################


def parse_args():
    parser = argparse.ArgumentParser(description="Infer examples using the Retriever")
    parser.add_argument("--index", type = str, default="fos_taxonomy_labels_02_embed",help = "index name")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use for inference")
    parser.add_argument("--log_path", type=str, default="src/metadata_mapper/logs", help="The path to the logs")
    parser.add_argument("--instruction", type=str, default="src/metadata_mapper/data/fos_taxonomy/fos_taxonomy_v0.1.2.json", help="The path to embedding model instruction")
    return parser.parse_args()

def main():
    args = parse_args()
    model_instruction = json.load(open(args.instruction, "r"))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    retriever = Retriever(
        ips = f"http://{ES_HOST}:{ES_PORT}",
        index=args.index,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        reranker_model="BAAI/bge-reranker-v2-m3",
        device=args.device,
        instruction=model_instruction['query_instruction'],
        cache_dir=MODEL_ARTEFACTS,
        log_path=args.log_path,
        es_passwd=ES_PASSWD
    )
    # infer some examples
    example_1 = "give me papers and publications about food and nutrition."
    for approach in ["cosine", "elastic"]:
        res = retriever.search_elastic (query=example_1, how_many=2, approach=approach)
        reranked_res = retriever.rerank_hits (query=example_1, hits=res["hits"])
        print (res)
        print (reranked_res)
    
if __name__ == "__main__":
    main()
    