# scinobo-taxonomy-mapper
This repository is used to map other taxonomies/keywords/phrases to the SciNoBo FoS taxonomy. 

Given a phrase or a sentence related to a Field of Science, the tool outputs the most similar 
L1/L2/L3/L4/L5/L6 FoS paths.

The tool can be used for identifying relevant FoS to a use case. Additionally, providing 
sentences like: "breast cancer in artificial intelligence" it will result in FoS paths relevant to 
"breast cancer" & "artificial intelligence", when using the `dense retrieval` functionality.

The tool uses `instructor-xl` embeddings and `dense retrieval` or `lexical retrieval`.

## Info

The server provides one functionality:

1. **Infer Mapper**: This functionality allows you to infer the most similar L1/L2/L3/L4/L5/L6 FoS paths given a user query. The endpoint for this functionality is `/infer_mapper`. The arguments provided to this endpoint are as follows:

```json
{
  "id": "an id, used for a reference point",
  "text": "the query that you want to search",
  "k": "how many retrieved results to return. Required",
  "approach": "<knn or elastic>" // the retrieval approach. When using "knn", then "dense retrieval" with cosine similarity is performed. When 
  // "elastic" is selected then "lexical retrieval" is performed, using BM25 in the "level_6" FoS field of the taxonomy
}
```

## Docker Build and Run

To build the Docker image, use the following command:

```bash
docker build -t scinobo-fos-mapper .
```

To run the Docker container, create an environment file with the following values:
- ES_PASSWD: the password to the elastic installation
- CA_CERTS_PATH: the path to the certification for elastic. This can be ommited if you have an elastic version < 8.x. Also omit it in the server run command.
- DEVICE: "cuda" or "cpu"
- ES_HOST: the ip of the elastic installation. Default: localhost
- ES_PORT: the port of the elastic installation. Default: 9200
- INDEX_NAME: the name of the elastic index, which hosts the taxonomy. Default: fos_taxonomy_01_embed
- FOS_TAXONOMY_VERSION: the version of the taxonomy. Default: v0.1.1

### Server:
```bash
docker run --env-file <path to the env file> -i -d --name scinobo-fos-mapper-docker --rm --gpus all -p 1990:1990 -v <path to the certs in host>:/certs/ scinobo-fos-mapper uvicorn fos_mapper.server.api:app --host 0.0.0.0 --port 1990
```

This will serve the uvicorn server of `api.py` which supports the `/infer_mapper` endpoint.