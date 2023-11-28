from abc import abstractmethod
from elasticsearch import Elasticsearch


class Source(object):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieval(self):
        pass

    class SourceDocBM25(Source):
        def __init__(self, es_host, es_port, index_name) -> None:
            self.es = Elasticsearch([{'host': es_host, 'port': es_port}])
            self.index_name = index_name

        def retrieval(self, query):
            search_body = {
                "query": {
                    "match": {
                        "content": query
                    }
                }
            }
            res = self.es.search(index=self.index_name, body=search_body)
            hits = res['hits']['hits']
            return [hit['_source']['content'] for hit in hits]
