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
        es_passwd,
        ca_certs_path=None
    ):
        self.device = device
        self.stop_ws = stopwords.words('english')
        self.stop_ws.extend(['may', 'could'])
        self.model = INSTRUCTOR(embedding_model, cache_folder=cache_dir, device=self.device)
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
        self.logger = logging.getLogger(__name__)
        
    def embed_query(self, query):
        return self.model.encode(query, show_progress_bar=True)
    
    def search_elastic_dense(self, query, how_many, approach="knn"):
        # embed the query using the instructor model
        query_emb = self.embed_query(
            [[self.instruction,query]]
        )
        if approach == "knn":
            # this approach is for the versions previous to 8.x
            res = self.es.search(
                index=self.index,
                body = {
                    "size": how_many,
                    "query": {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'fos_vector') + 1.0",
                                "params": {
                                    "query_vector": query_emb.tolist()[0]
                                }
                            }
                        }
                    }
                }
            )
            return res.body["hits"]["hits"]
        elif approach == "elastic":
            return self.search_elastic(query, how_many=how_many) # this simply searches by text
        elif approach == "cosine":
            raise NotImplementedError('Cosine is not implemented yet')
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
    
    def search_elastic(self, query, how_many=100):
        self.logger.debug('original query: {}'.format(query))
        qq = query
        for c in '~`@#$%^&*()_+=\'{}][:;"\\|΄<>,.?/': # this removes special chars, that we can specify
            qq = qq.replace(c, ' ')
        qq = ' '.join([w for w in qq.split() if w.lower() not in stop_ws][:900])
        self.logger.debug('processed query: {}'.format(qq))
        the_shoulds = [
            {
                "match": {
                    "level_6": {
                        "query": qq,
                        "boost": 1, 'minimum_should_match': "100%"
                    }
                }
            },
            {
                "match_phrase": {
                    "level_6": {
                        "query": qq,
                        "boost": 1, "slop": 2
                    }
                }
            },
            {
                "match_phrase": {
                    "level_6": {
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
                            "level_6": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "80%"
                            }
                        }
                    },
                    {
                        "match": {
                            "level_6": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "60%"
                            }
                        }
                    },
                    {
                        "match": {
                            "level_6": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "40%"
                            }
                        }
                    },
                    {
                        "match": {
                            "level_6": {
                                "query": qq,
                                "boost": 1, 'minimum_should_match': "30%"
                            }
                        }
                    },
                ]
            )
        bod     = {"size" : how_many, "query": {"bool": {"should": the_shoulds, "minimum_should_match": 1}}}
        ####################################################################################################
        res         = self.es.search(index=self.index, body=bod, request_timeout=120)
        ####################################################################################################
        return res["hits"]["hits"]

    def clean_sent(self, sent):
        return re.sub('\(.+?\)', '', sent)