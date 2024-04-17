"""
- This script contains the code to perform dense retrieval from the FoS taxonomy elasticsearch
- All the functionality is wrapped around a class called DenseRetriever
- The Denseretriever class must be able to connect to an elasticsearch instance

NOTE For now we do not pass any credentials. However, in the future this must change.

-- No args provided this will be called from the FastAPI server.
"""

import re
import torch
import logging

from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from InstructorEmbedding import INSTRUCTOR

###############################################################################################
stop_ws = stopwords.words('english')
stop_ws.extend(['may', 'could'])
###############################################################################################

class Retriever():
    def __init__(
        self, 
        ips, 
        index,  
        device,
        cache_dir,
        embedding_model,
        instruction,
        log_path,
        ca_certs_path,
        es_passwd
    ):
        self.device = device
        self.stop_ws = stopwords.words('english')
        self.stop_ws.extend(['may', 'could'])
        self.model = INSTRUCTOR(embedding_model, cache_folder=cache_dir, device=self.device)
        self.instruction = instruction
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
        self.logger = logging.getLogger(__name__)
        
    def embed_query(self, query):
        return self.model.encode(query, show_progress_bar=True)
    
    def search_elastic_dense(self, query, approach="knn"):
        # embed the query using the instructor model
        query_emb = self.embed_query(
            [[self.instruction,query]]
        )
        if approach == "knn":
            res = self.es.search(
                index=self.index,
                knn={"field": "fos_vector", "query_vector": query_emb.tolist(), "k": 10, "num_candidates": 100}
            )
            return res
        elif approach == "elastic":
            return self.search_elastic(query) # this simply searches by text
        elif approach == "cosine":
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['fos_vector']) + 1.0",
                        "params": {"query_vector": query_emb},
                    }
                }
            }
            res = self.es.search(
                index=self.index,
                body={
                    "size": 10,
                    "query": script_query
                }
            )
            return res
        else:
            raise NotImplementedError('Only knn, elastic and cosine are implemented')

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
    
    def search_elastic(self, query, how_many=100, token_threshold=200):
        self.logger.debug('original query: {}'.format(query))
        qq = query
        for c in '~`@#$%^&*()_+=\'{}][:;"\\|΄<>,.?/':
            qq = qq.replace(c, ' ')
        qq = ' '.join([w for w in qq.split() if w.lower() not in stop_ws][:900])
        self.logger.debug('processed query: {}'.format(qq))
        the_shoulds = [
            {
                "match": {
                    "text": {
                        "query": qq,
                        "boost": 1, 'minimum_should_match': "100%"
                    }
                }
            },
            {
                "match_phrase": {
                    "text": {
                        "query": qq,
                        "boost": 1, "slop": 2
                    }
                }
            },
            {
                "match_phrase": {
                    "text": {
                        "query": qq,
                        "boost": 1, "slop": 4
                    }
                }
            },
        ]
        if len(qq.split())>5:
            the_shoulds.extend(
                [
                    {
                        "match": {
                            "text": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "80%"
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "60%"
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "40%"
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "30%"
                            }
                        }
                    },
                ]
            )
        bod     = {"size" : how_many, "query": {"bool": {"should": the_shoulds, "minimum_should_match": 1}}}
        ####################################################################################################
        # res         = self.es.search(index=self.index, doc_type=self.doc_type, body=bod, request_timeout=120)
        res         = self.es.search(index=self.index, body=bod, request_timeout=120)
        res         = [
            [x['_source']['text'], x['_score'], x['_id']]
            for x in res['hits']['hits']
            if len(x['_source']['text'].split())<=token_threshold
        ]
        ####################################################################################################
        return res

    def clean_sent(self, sent):
        return re.sub('\(.+?\)', '', sent)