import logging
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever():
    """
    This class provides functionalities for performing dense and lexical retrieval from an Elasticsearch index. 
    It supports the retrieval of the most similar hits based on the following index types: "fos label", "venue names", and "affiliation".
    """    
    # NOTE For now we do not pass any credentials. However, in the future this must change.
    def __init__(
        self, 
        ips:str = None, 
        index:str= None,  
        device: str = "cuda",
        cache_dir: str = None,
        embedding_model:str= "nomic-ai/nomic-embed-text-v1.5",
        reranker_model:str = "BAAI/bge-reranker-v2-m3",
        instruction:str = None,
        log_path:str = None,
        es_passwd:str = None,
        ca_certs_path=None
    ):
        self.device = device
        self.stop_ws = stopwords.words('english') + ['may', 'could']
        self.embedding_model = SentenceTransformer(embedding_model, cache_folder=cache_dir, device=self.device, trust_remote_code=True)
        self.reranker_model = CrossEncoder (
            reranker_model, 
            cache_dir=cache_dir,
            automodel_args={"torch_dtype": "auto"}, 
            trust_remote_code=True)
        self.instruction = instruction
        if ca_certs_path is None:
            self.es = Elasticsearch(
                ips, 
                max_retries=10, 
                retry_on_timeout=True, 
                basic_auth=('elastic', es_passwd)
            )
        else:
            self.es = Elasticsearch(
                ips, 
                ca_certs=ca_certs_path, 
                max_retries=10, 
                retry_on_timeout=True, 
                basic_auth=('elastic', es_passwd)
            )
        self.index = index
        logging.basicConfig(
            #filename=f'{log_path}/retriever.log', # TODO add self.logger path
            level=logging.DEBUG,
            format="%(asctime)s; %(levelname)s; %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def identify_index_type(self):
        """
        Check index mapping field keys to identify the index type.
        """

        if not self.index:
            raise ValueError("Index is None. Please provide a valid index.")
                
        mapping = self.es.indices.get_mapping(index=self.index)
        properties = mapping[self.index]["mappings"]["properties"]
        expected_fields = ["fos_label", "affiliation", "venue_name"]

        for field in expected_fields:
            if field in properties:
                return field
        
        raise KeyError(
            f"Unexpected fields in mapping. Found keys: {list(properties.keys())}. "
            f"Expected one of: {expected_fields}."
        )
        
    def normalize_query (self, query):
        """Remove punctuations and stop words."""

        cleaned_query = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
        filtered_query = " ".join(
            [w for w in cleaned_query.split() if w.lower() not in self.stop_ws][:900]
        )
        return filtered_query
    
    def embed_query(self, query):
        """Embed query using the embedding model"""
        return self.embedding_model.encode(query, show_progress_bar=True)

    def search_elastic(self, query, how_many, approach="cosine"):
        """Run lexical or semantic search on the ES index."""

        query = self.normalize_query(query)
        
        if approach == "cosine":
            results = self.run_dense_search(query, how_many)
        elif approach == "elastic":
            results = self.run_lexical_search(query, how_many)
        elif approach == "hybrid":
            results = self.run_hybrid_search(query, how_many)
        else:
            raise NotImplementedError("Only 'cosine', 'elastic' and 'hybrid' search approaches are currently implemented") 
        results["index_type"] = self.identify_index_type()
        return results

    def run_dense_search(self, query, how_many=100):
        """Run semantic search based on dense vectors similarity."""

        query_emb = self.embed_query(self.instruction + query)
        # this approach is for the versions previous to 8.x
        res = self.es.search(
            index=self.index,
            body={
                "size": how_many,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_emb.tolist()  # Ensure it is a list
                            }
                        }
                    }
                }
            }
        )
        results = res.body["hits"]
        return results

    def run_hybrid_search (self, query, how_many=100):
        """Run a hybrid search combining both dense and lexical retrieval and reranking the results."""

        lexical_hits = self.run_lexical_search(query, how_many = how_many)["hits"]
        dense_hits = self.run_dense_search(query, how_many=how_many)["hits"]

        self.logger.debug("lexical_hits: {}".format(
            list({k: v for k, v in (hit.get("_source") or {}).items() if k != "vector"} for hit in lexical_hits)
        ))

        self.logger.debug("dense_hits: {}".format(
            list({k: v for k, v in (hit.get("_source") or {}).items() if k != "vector"} for hit in dense_hits)
        ))

        reranked_hits =  self.rerank_hits(query=query, hits=lexical_hits + dense_hits, how_many=how_many)
        return {"hits": reranked_hits}

    def run_lexical_search(self, query, how_many=100):
        """Run keyword search on given field name string."""

        # Define field_name to search 
        field_name = self.identify_index_type()
        normalized_query = self.normalize_query(query)

        self.logger.debug('Field name for search: {}'.format(field_name))
        self.logger.debug('original query: {}'.format(query))
        self.logger.debug('Normalized query: {}'.format(normalized_query))

        # Adjust min_should_match based on query length
        query_terms = len(normalized_query.split())
        if query_terms <= 3:
            min_should_match = "75%"
        elif query_terms <= 5:
            min_should_match = "50%"
        else:
            min_should_match = "30%"

        the_shoulds = [
            {
                "match": {
                    field_name: {
                        "query": normalized_query,
                        "boost": 1,
                        "minimum_should_match": min_should_match
                    }
                }
            },
            {
                "match_phrase": {
                    field_name: {
                        "query": normalized_query,
                        "boost": 1,
                        "slop": 5  # Allow up to 5 extra words between terms
                    }
                }
            },
            {
                "term": {
                    field_name: {
                        "value": normalized_query,
                        "boost": 2
                    }
                }
            }
        ]

        # Add additional clauses for long queries
        if query_terms > 5:
            the_shoulds.extend([
                {
                    "match": {
                        field_name: {
                            "query": normalized_query,
                            "boost": 1,
                            "minimum_should_match": "60%"
                        }
                    }
                },
                {
                    "match": {
                        field_name: {
                            "query": normalized_query,
                            "boost": 1,
                            "minimum_should_match": "40%"
                        }
                    }
                }
            ])

        bod = {
            "size": how_many,
            "query": {
                "bool": {
                    "should": the_shoulds,
                    "minimum_should_match": 1 
                }
            }
        }

        res = self.es.search(index=self.index, body=bod, request_timeout=120)
        results = res.body.get("hits", {})

        # Replace list elements with empty dicts in case < k hits are retrieved
        results["hits"].extend([{}] * (how_many - len(results["hits"])))
        
        return results

    def rerank_hits(self, query, hits, how_many):
        """Rerank retriever results using a cross-encoder Reranker. Return a sorted list of results based on the Reranker scores."""

        # Find the field for which we are going to rerank the hits
        field_name = self.identify_index_type()
        
        # Get documents from hits
        docs = [hit.get("_source", {}).get(field_name, "") for hit in hits]

        # If no hits are returned from the retriever, return an empty list
        if not hits:
            return []

        # Rerank retrieved hits
        reranked_results = self.reranker_model.rank(
            query=query,
            documents=docs,  
            return_documents=False, 
            convert_to_tensor=False,
            apply_softmax=True
        )

        if len(reranked_results) != len(hits):
            self.logger.warning("Mismatch between hits and reranked results length")

        # Assign reranker scores to hits
        for ranked_doc in reranked_results:
            index = ranked_doc["corpus_id"]
            reranker_score = ranked_doc["score"]
            hits[index]["_reranker_score"] = reranker_score

        # Sort hits based on reranker score
        sorted_hits = sorted(hits, key=lambda x: x.get("_reranker_score", float('-inf')), reverse=True)

        # Trim the list to the requested number of hits
        sorted_hits = sorted_hits[:how_many]

        self.logger.debug("reranked_hits: {}".format(
            [{k: v for k, v in hit.get("_source", {}).items() if k != "vector"} for hit in sorted_hits]
        ))

        return sorted_hits
