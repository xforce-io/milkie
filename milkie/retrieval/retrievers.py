from typing import List
from llama_index import VectorStoreIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import BaseRetriever, BM25Retriever
from llama_index.schema import NodeWithScore

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
            for vectorNode in vectorNodes:
                if vectorNode.score < 0.4:
                    continue

                theNode = nodeIdToNode.get(vectorNode.node_id)
                if theNode is None:
                    vectorNode.node.metadata["emb_score"] = vectorNode.score
                    nodes.append(vectorNode)
                    nodeIdToNode[vectorNode.node_id] = vectorNode
                else:
                    theNode.node.metadata["emb_score"] = vectorNode.score

        if self.sparseRetriever:
            bm25Nodes = self.sparseRetriever._retrieve(query_bundle)
            for bm25Node in bm25Nodes:
                theNode = nodeIdToNode.get(bm25Node.node_id)
                if theNode is None:
                    bm25Node.node.metadata["bm25_score"] = bm25Node.score
                    nodes.append(bm25Node)
                    nodeIdToNode[bm25Node.node_id] = bm25Node
                else:
                    theNode.node.metadata["bm25_score"] = bm25Node.score
        return nodes