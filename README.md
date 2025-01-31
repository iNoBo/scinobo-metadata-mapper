# scinobo-metadata-mapper

This repository provides a tool for mapping phrases to specific SciNoBo metadata structures, including:

1. **Field of Study (FoS) taxonomy**
2. **Publication venues**
3. **Affiliations**

Given a phrase or sentence related to a target metadata label, the tool returns the most relevant label using `dense`, `lexical`, or `hybrid` retrieval on an Elasticsearch index.


## Example Queries & Outputs  

**Input:** `"Cancer"`  
**Output:**  
```json
{
    "fos_label": "Oncology & Carcinogenesis",
    "fos_level": "level_3"
}
```

**Input:** `"Publications from INT J EDUC REV"`  
**Output:**  
```json
{
    "venue_name": "int j educ rev",
    "full_name": "International Journal of Educational Review"
}
```

**Input:** `"Papers from MIT"`  
**Output:**  
```json
{
    "affiliation": "MIT",
    "aff_type": "acronym",
    "full_name": "Massachusetts Institute of Technology"
}
```

## Info

The server provides 3 endpoints. One for each metadata category:

1. **/search/fos_taxonomy**

2. **/search/publication_venues**

3. **/search/affiliations**

The arguments provided to each endpoint are as follows:

```json
{
  "id": "an id, used for a reference point",
  "text": "the query that you want to search",
  "k": "how many retrieved results to return. Required",
  "approach": "<cosine, elastic, hybrid>", // the retrieval approach. When using "cosine", then "dense retrieval" with cosine similarity is performed. When 
  // "elastic" is selected then "lexical retrieval" is performed, using BM25 in the corresponding field name strig. When
  // "hybrid" is selected both dense and lexical retrieval is performed and a Cross-Encoder reranks the joined results list keeping the top_k most relevant results.
  "rerank": "Whether to rerank the retrieved list of results using a Cross-Encoder model."
}
```

## Docker Build and Run

To build the Docker image, use the following command:

```bash
docker build -t .
```

To run the Docker container, create an **environment file** (`.env`) with the following values:  

### **Required Variables**  
- `DEVICE`: `"cuda"` or `"cpu"` (sets the processing device)  
- `ES_HOST`: The IP of the Elasticsearch installation (default: `localhost`)  
- `ES_PORT`: The port of the Elasticsearch installation (default: `9200`)  
- `ES_PASSWD`: The password for the Elasticsearch installation  
- `MAPPER_HOST`: The IP address to run the server  
- `MAPPER_PORT`: The port to run the server  

### **Optional Variables (for Evaluation Scripts)**  
If you want to execute the evaluation scripts, include the following values:  
- `OLLAMA_HOST`: The host for Ollama  
- `OLLAMA_PORT`: The port for Ollama  
- `LANGFUSE_PUBLIC_KEY`: The public key for Langfuse  
- `LANGFUSE_SECRET_KEY`: The secret key for Langfuse  
- `LANGFUSE_HOST`: The Langfuse host URL  

### **Additional Notes**  
- To run the evaluation scripts, you must uncomment the `--network=text2sql_scinobo-network` flag in the `devcontainer` configuration.  
- This allows access to the network and enables Ollama for dataset generation.  

### Server:
```bash
docker run --env-file <path to the env file> -i -d --name scinobo-metadata-mapper-docker --rm --gpus all -p 1990:1990 -v <path to the certs in host>:/certs/ scinobo-metadata-mapper uvicorn src.metadata_mapper.server.api:app --host 0.0.0.0 --port 1990
```

This will serve the uvicorn server of `api.py` which supports the 3 endpoints.