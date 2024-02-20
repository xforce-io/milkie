import logging
from typing import List
from llama_index import VectorStoreIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import BaseRetriever, BM25Retriever
from llama_index.schema import NodeWithScore

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    def __init__(
            self, 
            denseRetriever :VectorStoreIndex, 
            sparseRetriever :BM25Retriever):
        self.denseRetriever = denseRetriever
        self.sparseRetriever = sparseRetriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        #combine two nodes, and record the score in the metadata
        nodes = []
        nodeIdToNode = {}
        if self.denseRetriever:
            vectorNodes = self.denseRetriever._retrieve(query_bundle)
            logger.debug(f"dense_retriever_recall_num[{len(vectorNodes)}]")
            for vectorNode in vectorNodes:
                if vectorNode.score < 0.4:
                    continue

                theNode = nodeIdToNode.get(vectorNode.node_id)
                if theNode is None:
                    nodes.append(vectorNode)
                    nodeIdToNode[vectorNode.node_id] = vectorNode

        if self.sparseRetriever:
            bm25Nodes = self.sparseRetriever._retrieve(query_bundle)
            logger.debug(f"sparse_retriever_recall_num[{len(bm25Nodes)}]")
            for bm25Node in bm25Nodes:
                theNode = nodeIdToNode.get(bm25Node.node_id)
                if theNode is None:
                    nodes.append(bm25Node)
                    nodeIdToNode[bm25Node.node_id] = bm25Node
        return nodes