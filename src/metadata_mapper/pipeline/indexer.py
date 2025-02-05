from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch

class Indexer:
    """"Handle Elasticsearch indexing."""
    def __init__(
        self,
        index_name,
        es_passwd,
        ips,
        mapping=None,
        batch_size=1000,
        delete_index=False,
        mapping_type=None,
        ca_certs_path=None):

        if ca_certs_path is None:
            self.es = Elasticsearch(
                ips, 
                max_retries=10, 
                retry_on_timeout=True,
                request_timeout=150,
                basic_auth=('elastic', es_passwd)
            )
        else:
            self.es = Elasticsearch(
                ips, 
                ca_certs=ca_certs_path, 
                max_retries=10, 
                retry_on_timeout=True, 
                request_timeout=150,
                basic_auth=('elastic', es_passwd)
            )
            
        self.index_name = index_name
        self.mapping = mapping
        self.mapping_type = mapping_type
        self.actions = []
        self.batch_size = batch_size
        
        # check if index exists
        if not self.check_if_index_exists():
            self.create_index()
        else:
            print(f'Index {self.index_name} already exists')
            if delete_index:
                self.delete_index()
                self.create_index()
            else:
                print(f'Index {self.index_name} already exists, force delete it first if you want, by parsing argument --delete_index True in indexer class')
    
    def delete_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        
    def check_if_index_exists(self):
        return self.es.indices.exists(index=self.index_name)
    
    def create_index(self):
        if self.mapping is None:
            raise Exception('No mapping provided')
        self.es.indices.create(index=self.index_name, body=self.mapping)
        
    def create_an_action(self, dato, op_type, the_id=None):
        if the_id is None:
            pass
        else:
            dato['_id'] = the_id
        dato['_op_type'] = op_type
        dato['_index'] = self.index_name
        return dato
    
    def upload_to_elk(self, finished=False):
        if (len(self.actions) >= self.batch_size) or (len(self.actions) > 0 and finished):
            flag = True
            while flag:
                try:
                    _ = bulk(self.es, iter(self.actions))
                    # pprint(result)
                    flag = False
                except Exception as e:
                    print(e)
                    if 'ConnectionTimeout' in str(e) or 'Connection timed out' in str(e):
                        print('Retrying')
                    else:
                        flag = False
            self.actions = []
    
    def process_folder(self, data):
        for dato in data:
            self.process_one_dato(dato)
            
    def process_one_dato(self, doc, op_type, the_id=None):
        self.actions.append(self.create_an_action(
            dato=doc, 
            op_type=op_type,
            the_id=the_id
        ))
        self.upload_to_elk(finished=False)
