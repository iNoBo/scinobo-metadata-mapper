"""
- This script contains the code to perform dense retrieval from the FoS taxonomy elasticsearch
- All the functionality is wrapped around a class called DenseRetriever
- The Denseretriever class must be able to connect to an elasticsearch instance

NOTE For now we do not pass any credentials. However, in the future this must change.

-- No args provided this will be called from the FastAPI server.
"""

import logging

from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever():
    def __init__(
        self, 
        ips, 
        index,  
        device,
        cache_dir,
        embedding_model,
        reranker_model,
        instruction,
        log_path,
        es_passwd,
        ca_certs_path=None
    ):
        self.device = device
        self.stop_ws = stopwords.words('english')
        self.stop_ws.extend(['may', 'could'])
        self.embedding_model = SentenceTransformer(embedding_model, cache_folder=cache_dir, device=self.device, trust_remote_code=True)
        self.reranker_model = CrossEncoder (reranker_model, cache_dir=cache_dir, automodel_args={"torch_dtype": "auto"}, trust_remote_code=True)
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
            filename=f'{log_path}/retriever.log', # TODO add self.logger path
            level=logging.DEBUG,
            format="%(asctime)s;%(levelname)s;%(message)s"
        )
        self.index_type = self.define_index_mapping_type()
        self.logger = logging.getLogger(__name__)
        

    def define_index_mapping_type (self):
        """Validate index mapping keys. In our case, an index can either include full taxonomy levels or single FoS labels."""

        mapping = self.es.indices.get_mapping(index=self.index)
        properties = mapping[self.index]["mappings"]["properties"]
        type_1_keys = {"fos_label", "level", "fos_vector"}
        type_2_keys = {
            "fos_vector", "level_1", "level_2", "level_3", "level_4", 
            "level_4_id", "level_5", "level_5_id", "level_5_name", "level_6"
        }
        index_keys = properties.keys()
        if index_keys == type_1_keys:
            return "fos_label"
        elif index_keys == type_2_keys:
            return "full_taxonomy"
        else:
            raise KeyError(
                f"Unexpected keys in mapping. Found keys: {index_keys}. "
                f"Expected either: {type_1_keys} or {type_2_keys}."
            )
        
    def normalize_query (self, query):
        """Remove punctuations and stop words."""

        for c in '~`@#$%^&*()_+=\'{}][:;"\\|΄<>,.?/':
            query = query.replace(c, ' ')
        query = ' '.join([w for w in query.split() if w.lower() not in self.stop_ws][:900])
        return query
    
    def embed_query(self, query):
        return self.embedding_model.encode(query, show_progress_bar=True)
    
    def search_elastic(self, query, how_many, approach="cosine"):
        """Run lexical or semantic search on the ES index."""

        query = self.normalize_query(query)
        query_emb = self.embed_query(self.instruction + query)

        if approach == "cosine":
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
                                "source": "cosineSimilarity(params.query_vector, 'fos_vector') + 1.0",
                                "params": {
                                    "query_vector": query_emb.tolist()  # Ensure it is a list
                                }
                            }
                        }
                    }
                }
            )
            results = res.body["hits"]
            results["index_type"] = self.index_type
            return results
        
        elif approach == "elastic":
            return self.run_lexical_search(query,  how_many=how_many) # this simply searches by text
        else:
            raise NotImplementedError('Only cosine and elastic search approaches are implemented')


    def run_lexical_search(self, query, how_many=100):
        """Run lexical search on given field name in an ES index."""

        # Define the field name to search.
        if self.index_type == "full_taxonomy":
            field_name = "level_6"
        elif self.index_type == "fos_label":
            field_name = self.index_type

        self.logger.debug('original query: {}'.format(query))
        qq = self.normalize_query(query)
        
        self.logger.debug('processed query: {}'.format(query))
        the_shoulds = [
            {
                "match": {
                    field_name: {
                        "query": qq,
                        "boost": 1, 'minimum_should_match': "100%"
                    }
                }
            },
            {
                "match_phrase": {
                    field_name: {
                        "query": qq,
                        "boost": 1, "slop": 2
                    }
                }
            },
            {
                "match_phrase": {
                    field_name: {
                        "query": qq,
                        "boost": 1, "slop": 4
                    }
                }
            },
        ]
        if len(qq.split())>5: # if the phrase is large enough
            the_shoulds.extend(
                [
                    {
                        "match": {
                            field_name: {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "80%"
                            }
                        }
                    },
                    {
                        "match": {
                            field_name: {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "60%"
                            }
                        }
                    },
                    {
                        "match": {
                            field_name: {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "40%"
                            }
                        }
                    },
                    {
                        "match": {
                            field_name: {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "30%"
                            }
                        }
                    },
                ]
            )
        bod = {"size" : how_many,  "query": {"bool": {"should": the_shoulds, "minimum_should_match": 1}}}
        res = self.es.search(index=self.index, body=bod, request_timeout=120)
       
        results = res.body["hits"]
        results["index_type"] = self.index_type
        return results

    def rerank_hits(self, query, hits):
        """Rerank retrieved results using a Cross Encoder model. Return a sorted list of results with updated scores."""
        
        docs = []  

        for hit in hits:
            if self.index_type == "fos_label":
                doc = hit["_source"]["fos_label"]
            else:
                doc = f"{hit['_source']['level_1']}/{hit['_source']['level_2']}/{hit['_source']['level_3']}/{hit['_source']['level_4']}/{hit['_source']['level_5_name']}"
            docs.append(doc)

        # Use the Cross Encoder to get reranker scores
        reranked_hits = self.reranker_model.rank(
            query=query,
            documents=docs,
            return_documents=True, 
            convert_to_tensor=False,
            apply_softmax=True
        )
        for i, hit in enumerate(reranked_hits):
            hits[i]["_reranker_score"] = hit["score"]

        sorted_hits = sorted(hits, key=lambda x: x["_reranker_score"], reverse=True)

        return sorted_hits

 