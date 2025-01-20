# Script to run an evaluation experiment on given query to fos_label dataset.
from langfuse import Langfuse
import requests
from dotenv import load_dotenv
import os
import json
import argparse
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

###################################################
load_dotenv()
MAPPER_HOST = os.environ["MAPPER_HOST"]
MAPPER_PORT = os.environ["MAPPER_PORT"]
##################################################

# init langfuse client
langfuse = Langfuse()

def calculate_reciprocal_rank (hits:dict, expected_output:dict):
  """
    Calculate the reciprocal rank for a list of hits returned by the retrieval process.

    The higher the golden label is ranked in the list, the higher its ranking score.
    If the golden label is not present in the list, the reciprocal rank is 0.0.
  """
  golden_label = expected_output["fos_label"].lower().strip()
  for i, hit in enumerate (hits, start=1):
    predicted_label = hit["fos_label"].strip()
    if predicted_label == golden_label:
      return 1/i

  return 0.0

def calculate_mrr_per_level(dataset_items, scores):
    """
    Calculates MRR per FoS level.
    """
    levels = {"level_1": [], "level_2": [], "level_3": [], "level_4": [], "level_5": [], "total": []}
    for item, score in zip(dataset_items, scores):
      level = item.expected_output.get("level")
      if not level:
          print(f"Warning: Missing 'level' in expected_output for item: {item}")
      elif level not in levels:
          print(f"Warning: Unexpected level '{level}' in item: {item}")
      else:
          levels[level].append(score)
      levels["total"].append (score)
    
    mrr_scores= {}
    for level, scores in levels.items():
       mrr_scores[level] = np.mean(scores) if scores else 0
       
    return mrr_scores

def plot_mrr_scores (ks, results, save_dir):
    """
    Plot MRR to k subplots for each FoS level.
    """
    levels = list(results.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(20, 5), sharey=True)
    fig.suptitle("MRR Scores by Level")
    
    for i, level in enumerate(levels):
        axes[i].plot(ks, results[level], label=level)
        axes[i].set_title(level)
        axes[i].set_xlabel("k")
        axes[i].set_ylabel("MRR")
        axes[i].set_ylim (0,1)
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    save_path = os.path.join(
       save_dir,
      "mrr_scores_per_level.png"
    )
    plt.savefig(save_path)

def plot_scores_distribution(all_scores, all_labels, save_dir):
    """Create density plots of confidence scores for both irrelevant and relevant retrieved labels across all top-k values."""

    k_values = range(1, len(all_scores) + 1)  # 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.4)

    for i, k in enumerate(k_values):
        scores = np.array(all_scores[k - 1])  # Shape: [queries, k]
        labels = np.array(all_labels[k - 1])  # Shape: [queries, k]

        # Flatten for plotting
        scores = scores.flatten()
        labels = labels.flatten()

        # Separate scores
        relevant_scores = scores[labels == 1]
        irrelevant_scores = scores[labels == 0]

        # Density functions
        sns.histplot(relevant_scores, color='blue', label='Relevant', kde=True, stat="density", bins=30, ax=axes[i])
        sns.histplot(irrelevant_scores, color='red', label='Irrelevant', kde=True, stat="density", bins=30, ax=axes[i])
        axes[i].set_title(f'Density Plot for Top-{k}')
        axes[i].set_xlabel('Confidence Score')
        axes[i].legend()

    fig.suptitle('Confidence Score Distributions for Relevant and Irrelevant Labels', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_scores_dist.png"))

def run_experiment(
    dataset_name: str,
    approach: str,
    save_dir:str,
    rerank:bool = True,
    only_locally:bool=False
):
    """
    Evaluate the taxonomy mapper on a Langfuse dataset for multiple values of k (predefined ks: 1 to 10).
    Results are stored in langfuse server and plots are saved locally.
    """
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    # Dataset
    dataset = langfuse.get_dataset(dataset_name)

    # expirement with different k values (1 to 10)
    ks = range(1, 11)
    all_scores = []
    all_labels = []
    
    # store MRR scores
    results = {f"level_{i}": [] for i in range(1, 6)}
    results["total"] = []  

    for k in ks:
        k_scores = [] # condifence scores for top-k documents (for all dataset items)
        k_labels = [] # ground truth labels (0 or 1) for top-k documents (for all dataset items)
        k_reciprocal_ranks = [] # rr scores (for all dataset items)
        experiment_name = f"mrr@{k}__{approach}_reranker_{rerank}"

        for item in tqdm(dataset.items, desc=f"Calculating MRR@{k} on given dataset..."):
            with item.observe(run_name=experiment_name) as trace_id:
                # Retrieve top-k results
                request_body = json.dumps({
                    "id": "string",
                    "text": item.input["query"],
                    "k": k,
                    "approach": approach,
                    "rerank": rerank
                })

                response = requests.post(
                    url=f"http://{MAPPER_HOST}:{MAPPER_PORT}/infer_mapper",
                    data=request_body,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code != 200:
                    raise ValueError(f"Failed to retrieve results: {response.status_code}, {response.text}")
                
                # Parse API response
                hits = response.json().get("retrieved_results", [])
                if rerank:
                  scores = [hit.get("reranker_score", 0) for hit in hits]
                else:
                   scores = [hit.get("score", 0) for hit in hits]
                labels = [1 if hit.get("fos_label","").strip().lower() == item.expected_output["fos_label"].strip().lower() else 0 for hit in hits]

                # Append scores and labels for current dataset item
                k_scores.append(scores)  
                k_labels.append(labels)
                
                # Calculate reciprocal rank for current dataset item
                reciprocal_rank = calculate_reciprocal_rank(hits, item.expected_output)
                k_reciprocal_ranks.append(reciprocal_rank)
                
                enumerated_hits = {}
                retriever_scores = {}
                reranker_scores = {}

                for i, hit in enumerate (hits, start=1):
                  enumerated_hits[i] = (hit.get("fos_label"), hit.get("level"))
                  retriever_scores[i] = hit.get ("score")
                  reranker_scores[i] = hit.get("reranker_score")
                
                # Upload results in langfuse
                if not only_locally:
                  langfuse.score(
                    trace_id=trace_id,
                    name="Results",
                    value=str(enumerated_hits)
                  )
                  langfuse.score(
                    trace_id=trace_id,
                    name="Reciprocal_Rank",
                    value=reciprocal_rank
                  )
                  langfuse.score(
                    trace_id=trace_id,
                    name="Retriever_score",
                    value=str(retriever_scores)
                  )
                  langfuse.score(
                    trace_id=trace_id,
                    name="Reranker_score",
                    value=str(reranker_scores)
                  )
        
        all_scores.append(np.array(k_scores))  # Shape: [queries, k]
        all_labels.append(np.array(k_labels))  # Shape: [queries, k]

        # Calculate MRR per level
        avg_mrr_per_level = calculate_mrr_per_level(dataset.items, k_reciprocal_ranks)
        for level, avg_mrr in avg_mrr_per_level.items():
            if level in results:
                results[level].append(avg_mrr)
    
    plot_mrr_scores(ks = ks , results=results, save_dir = save_dir)
    plot_scores_distribution(all_scores, all_labels, save_dir=save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run experiments using Langfuse datasets.")
  parser.add_argument(
      "--dataset", 
      required=True, 
      help="Langfuse dataset name."
  )
  parser.add_argument(
      "--approach", 
      required=False, 
      choices=["elastic", "cosine"], 
      default="cosine",
      help="Retrieval approach to use."
  )
  parser.add_argument(
     "--l",
     action="store_true",
     help="Run evaluation only locally."
  )
  parser.add_argument(
      "--rerank", 
      action="store_true",
      help="Whether to rerank the results or not."
  )
  parser.add_argument(
     "--save_dir",
     required=False,
     default= os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_results"),
     help="Directory to save evaluation results."
  )
  args = parser.parse_args()
  
  run_experiment(dataset_name=args.dataset, approach=args.approach, save_dir=args.save_dir, only_locally = args.l, rerank=args.rerank)
  langfuse.flush()