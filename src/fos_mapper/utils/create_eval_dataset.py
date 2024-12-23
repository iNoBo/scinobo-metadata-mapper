# Script to generate a synthetic query-to-FoS (Field of Science) dataset for evaluating the Taxonomy Mapper.

import os
import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
import ast

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from langfuse import Langfuse
 

###################################################

#DATA_PATH = os.path.join(importlib.resources.files(__package__.split(".")[0])._paths[0], "data")
DATA_PATH = "/workspaces/scinobo-taxonomy-mapper/src/fos_mapper/data"

###################################################

def get_fos_labels_hierarchy(fos_taxonomy_version="v0.1.2", max_labels_per_level: dict = {1: None, 2: 3, 3: 3, 4: 3, 5: 3}):
    """
    Get the Field of Study (FoS) names hierarchy in the form of nested dictionaries.
    """
    
    hierarchy = {}
    with open(os.path.join(DATA_PATH, f"fos_taxonomy_{fos_taxonomy_version}.json")) as fp:
        fos_taxonomy_data = json.load(fp)
    
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
            data.append({"label": key, "level": f"level_{current_level}"})
            # Add to the stack only if within the level limit (exclude level 6 and level 5 topics)
            if current_level < 5 and isinstance(value, dict):
                to_process.append((value, current_level + 1))               
    df = pd.DataFrame(data)
    if drop_na:
        df = df [df["label"] != "N/A"].reset_index(drop=True)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["label"], keep="first").reset_index(drop=True)
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

def generate_queries (fos_labels, pipeline):
    """Given a list of FoS labels, generate user queries"""

    queries = []
    for fos in tqdm(fos_labels, desc= "Generating user queries for given FoS labels..."):
        result = pipeline.run({"prompt_builder": {"query": fos}})
        queries.append (result["llm"]["replies"][0].strip ("'"))
    return queries



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fos_taxonomy_version", type=str, required=False, help="Version of the fos taxonomy.", default="v0.1.2")
    parser.add_argument("--prompt_name", type=str, required=False, help= "Langfuse stored prompt name to retrieve for dataset generation.", default="fos_label_evaluation_dataset_generation_prompt_v1.0")
    parser.add_argument("--model_name", type=str, required=False, help="LLM name for prompting.", default="llama3.2:3b")
    parser.add_argument("--max_labels_per_level", type=str, required=False, help="A dictionary (as a string) specifying the maximum number of FoS labels to include in the dataset at each level.", default= "{1: None, 2: 3, 3: 3, 4: 3, 5: 3}")
    parser.add_argument("--drop_na", type=str, required=False, help="Whether to drop rows with N/A values from the final dataset.", default=True)
    parser.add_argument("--drop_duplicates", type=str, required=False, help="Whether to drop duplicate FoS labels in the dataset.", default=True)
    args = parser.parse_args()

    # Create a hierarchy for fos taxonomy including only the fos labels (provide a max labels per level limitation if needed).
    hierarchy = get_fos_labels_hierarchy (fos_taxonomy_version = args.fos_taxonomy_version,  max_labels_per_level=ast.literal_eval(args.max_labels_per_level))

    # Create a dataframe from generated hierarchy.
    df = hierarchy_to_dataframe (hierarchy=hierarchy, drop_duplicates=args.drop_duplicates, drop_na=args.drop_na)

    # Initialize langfuse client
    langfuse = Langfuse()
    # Retrieve prompt from langfuse
    prompt = langfuse.get_prompt(args.prompt_name)

    # Generate queries given the selected FoS labels and create a column in the dataframe.
    llm_pipe = init_query_generator(model_name=args.model_name, prompt=prompt)
    df["query"] = generate_queries(df["FoS_label"], llm_pipe)
    df.reset_index(drop=True)
    
    # Store generated dataset to langfuse for later evaluation.
    local_items =  [

        {
            "input": {"query": row["query"]},
            "expected_output": {
                "fos_label": row["fos_label"],
                "level": row["level"]
            }
        }
        for _, row in df.itterows()
    ]

    langfuse.create_dataset(
        name="fos_labels_eval_dataset",
        description="Instances of user query and a single corresponding FoS label.",
        metadata={
            "author": "Panos",
            "date": "2024-12-23",
            "type": "benchmark"
        }
    )

    for item in local_items:
        langfuse.create_dataset_item(
            dataset_name="fos_labels_eval_dataset",
            input=item["input"],
            expected_output=item["expected_output"]
        )

    # Save dataset locally as well
    with open (os.path.join (DATA_PATH, f"fos_labels_hierarchy_{args.fos_taxonomy_version}.json"), "w") as fp:
        json.dump (hierarchy, fp, ensure_ascii=False, indent=2)
    
    with open (os.path.join (DATA_PATH, f"fos_eval_dataset_{args.fos_taxonomy_version}.csv"), "w") as fp:
        df.to_csv(fp, index=False)


if __name__ == "__main__":
    main()