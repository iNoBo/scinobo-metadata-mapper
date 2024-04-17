""" 

This script is used to infer a couple of examples.
It uses the inference.py script to perform dense retrieval

"""

# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
import json
import argparse

from fos_mapper.pipeline.inference import Retriever
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser(description="Infer examples using the dense retriever")
    parser.add_argument("--es_passwd", type=str, required=True, help="The password to the elastic installation")
    parser.add_argument("--ca_certs_path", type=str, required=True, help="The path to the ca certs")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use for inference")
    parser.add_argument("--es_host", type=str, default="localhost", help="The host of the elastic installation")
    parser.add_argument("--es_port", type=str, default="9200", help="The port of the elastic installation")
    parser.add_argument("--log_path", type=str, default="src/fos_mapper/logs", help="The path to the logs")
    parser.add_argument("--model_artefacts_path", type=str, default="src/fos_mapper/model_artefacts", help="The path to the model artefacts")
    parser.add_argument("--fos_taxonomy_instruction", type=str, default="src/fos_mapper/data/fos_taxonomy_instruction_0.1.0.json", help="The path to the fos taxonomy instruction")
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    es_passwd = args.es_passwd
    ca_certs_path = args.ca_certs_path
    device = args.device
    es_host = args.es_host
    es_port = args.es_port
    log_path = args.log_path
    model_artefacts_path = args.model_artefacts_path
    fos_taxonomy_instruction_path = args.fos_taxonomy_instruction
    # load the taxonomy
    fos_taxonomy_instruction = json.load(open(fos_taxonomy_instruction_path, "r"))
    # instantiate the retriever
    retriever = Retriever(
        ips = [
            f"https://{es_host}:{es_port}"
        ],
        index="fos_taxonomy_01_embed",
        embedding_model="hkunlp/instructor-xl",
        device=device,
        instruction=fos_taxonomy_instruction['query_instruction'],
        cache_dir=model_artefacts_path,
        log_path=log_path,
        ca_certs_path=ca_certs_path,
        es_passwd=es_passwd
    )
    # infer some examples
    example_1 = "breast cancer in artificial intelligence"
    res = retriever.search_elastic_dense(example_1, "knn")
    pprint(res)
    
    
if __name__ == "__main__":
    main()