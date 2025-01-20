# Script to generate a synthetic query-to-FoS dataset for evaluation purposes.

import os
import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
import ast
import random
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from langfuse import Langfuse
from dotenv import load_dotenv
import os
from utils.data_handling import load_json

###################################################
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
LANGFUSE_PUBLIC_KEY= os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY= os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = os.environ["LANGFUSE_HOST"]
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
OLLAMA_PORT = os.environ["OLLAMA_PORT"]
###################################################

def get_fos_labels_hierarchy(fos_taxonomy_data, max_labels_per_level: dict = {1: None, 2: 3, 3: 3, 4: 3, 5: 3}):
    """
    Get the Field of Study (FoS) names hierarchy in the form of nested dictionaries.
    """
    
    hierarchy = {}
    
    for d in fos_taxonomy_data:
        l1 = d["level_1"]
        l2 = d["level_2"]
        l3 = d["level_3"]
        l4 = d["level_4"]
        l5_name = d["level_5_name"] 
        l5_topics = d["level_5"]    
        l6_topics = d["level_6"]    
    
        if l1 not in hierarchy:
            hierarchy[l1] = {}
        if l2 not in hierarchy[l1]:
            hierarchy[l1][l2] = {}
        if l3 not in hierarchy[l1][l2]:
            hierarchy[l1][l2][l3] = {}
        if l4 not in hierarchy[l1][l2][l3]:
            hierarchy[l1][l2][l3][l4] = {}
        if l5_name not in hierarchy[l1][l2][l3][l4]:
            hierarchy[l1][l2][l3][l4][l5_name] = {       
                "l5_topics": [topic.strip() for topic in l5_topics.split("----")],    
                "l6_topics": []
            }
        hierarchy[l1][l2][l3][l4][l5_name]["l6_topics"].extend(
            topic.strip() for topic in l6_topics.split("----")
        )
    
    # Apply limits to each level
    if max_labels_per_level:
        # Create a list of levels to process
        to_process = [(hierarchy, 1)]  # Start with level 1
        # Process each level until the list is empty
        while to_process:
            current_hierarchy, level = to_process.pop(0)  # Get the first item to process
            if not isinstance(current_hierarchy, dict):
                # If it's not a dictionary (e.g., list of topics), skip processing
                continue
            # Apply limit to the current level
            if level in max_labels_per_level and max_labels_per_level[level] is not None:
                keys = list(current_hierarchy.keys())
                if len(keys) > max_labels_per_level[level]:
                    # Keep only a limited number of keys
                    selected_keys = random.sample(keys, max_labels_per_level[level])
                    # Remove keys that were not selected
                    for key in keys:
                        if key not in selected_keys:
                            del current_hierarchy[key]
            # Add sub-levels to the list for further processing
            for key in current_hierarchy:
                to_process.append((current_hierarchy[key], level + 1))
    return hierarchy


def hierarchy_to_dataframe(hierarchy:dict, drop_na:bool=True, drop_duplicates:bool=True):
    """
    Convert a nested hierarchy dictionary into a df. Stops processing at level 5.
    """

    data = []  # To store labels and their levels
    to_process = [(hierarchy, 1)]  # Start with  level 1
    while to_process:
        current_hierarchy, current_level = to_process.pop()  # Get the next item to process
        for key, value in current_hierarchy.items():
            # Add the current key and its level to the data
            data.append({"fos_label": key, "level": f"level_{current_level}"})
            # Add to the stack only if within the level limit (exclude level 6 and level 5 topics)
            if current_level < 5 and isinstance(value, dict):
                to_process.append((value, current_level + 1))               
    df = pd.DataFrame(data)
    if drop_na:
        df = df [df["fos_label"] != "N/A"].reset_index(drop=True)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["fos_label"], keep="first").reset_index(drop=True)
    return df 


def init_query_generator (model_name:str=None, prompt:str=None):
    """Initialize a pipeline to run the llm"""

    llm = OllamaGenerator(
                model=model_name,
                url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
                generation_kwargs = {"temperature": 0.6, "top_k": 40, "top_p": 0.8}
            )
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=prompt))
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder", "llm")
    return pipe

def generate_queries (labels, pipeline):
    """Given a list of FoS labels, generate user queries"""

    queries = []
    for fos in tqdm(labels, desc= "Generating user queries for given FoS labels..."):
        result = pipeline.run({"prompt_builder": {"query": fos}})
        queries.append (result["llm"]["replies"][0].strip ("'"))
    return queries

def create_fos_labels_dataset (data, llm_pipe):
    """Create a dataset with synthetic queries and corresponding fos labels"""
    
    # Create a hierarchy for fos taxonomy including only the fos labels (provide a max labels per level limitation if needed).
    hierarchy = get_fos_labels_hierarchy (fos_taxonomy_data=data)
    # Create a dataframe from generated hierarchy.
    df = hierarchy_to_dataframe (hierarchy=hierarchy, drop_duplicates=True, drop_na=True)
    df["query"] = generate_queries(df["fos_label"], llm_pipe)
    df.reset_index(drop=True)
    return df


def create_venue_names_dataset (data, ll_pipe, num_of_examples):

    data_for_df = []
    for d in data:       
        alt_names = list (set (d.get("alternate_names", []))
        if alt_names:
            d ["label"] = random.choice(alt_names) # get one random alt name
            data_for_df.append (d)
    # create dataset examples
    examples  = random.sample (data_for_df, k=min(num_of_examples, len(data_for_df)))
    df = pd.DataFrame (examples)
    # generate queries for given venue name labels
    df["query"] = generate_queries (df ["label"], ll_pipe)
    df.reset_index(drop=True)
    return df 

def create_affiliation_dataset (data, llm_pipe, num_of_examples):
    return
        
def save_dataset (langfuse_instance, dataset, dataset_name, dataset_description):

    # Store generated dataset to langfuse for monitoring.
    local_items =  [
        {
            "input": {"query": row["query"]},
            "expected_output": {column: row[column] for column in dataset.columns() if column != "query" }
        }
        for _, row in dataset.iterrows()
    ]
    langfuse.create_dataset(
        name=dataset_name,
        description=dataset_description
    )
    for item in local_items:
        langfuse.create_dataset_item(
            dataset_nam=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"]
    )
    with open (os.path.join (DATA_PATH, f"datasets/{dataset_name}.csv"), "w") as fp:
        dataset.to_csv(fp, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_prefix", type=str, required=True, help="")
    parser.add_argument("--version", type=str, required=True, help="Version of the input data file.")
    parser.add_argument("--prompt_name", type=str, required=True, help= "Langfuse stored prompt name to retrieve for dataset generation.")
    parser.add_argument("--prompt_version", type=str, required=True, help="")
    parser.add_argument("--model_name", type=str, required=False, help="LLM name for prompting.", default="llama3.2:3b")
    args = parser.parse_args()

    # Initialize langfuse client
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
        )
    
    # load prompt
    prompt_template = langfuse.get_prompt(args.prompt_name, version= args.prompt_version, label='')
    # init pipeline
    llm_pipe = init_query_generator(model_name=args.model_name, prompt=prompt_template.prompt)
    # load data
    data = load_json (os.path.join(DATA_PATH, f"{args.data_file_prefix}/{args.data_file_prefix}_{args.version}.json"))
    # create_dataset 
    if args.data_file_prefix == "fos_taxonomy":
        dataset = create_fos_labels_dataset(data = data, llm_pipe = ll_pipe)
    elif args.data_file_prefix == "publication_venues":
        dataset = create_venue_names_dataset(data = data, llm_pipe = ll_pipe)
    else:
        raise ValueError ("Only fos_taxonomy and publication_venues are currently supported.")

if __name__ == "__main__":
    main()