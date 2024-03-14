import logging
from typing import List
from llama_index import VectorStoreIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import BaseRetriever, BM25Retriever
from llama_index.schema import NodeWithScore
from llama_index.indices.utils import truncate_text

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
            for node in vectorNodes:
                fmtText = truncate_text(node.node.text, 100).replace("\n", "//")
                logger.debug(f"score[{node.score:.2f}] content[{fmtText}]")
            
            for vectorNode in vectorNodes:
                if vectorNode.score < 0.4:
                    continue

                theNode = nodeIdToNode.get(vectorNode.node_id)
                if theNode is None:
                    nodes.append(vectorNode)
                    nodeIdToNode[vectorNode.node_id] = vectorNode
                    vectorNode.metadata["rrf"] = False

        if self.sparseRetriever:
            bm25Nodes = self.sparseRetriever._retrieve(query_bundle)
            logger.debug(f"sparse_retriever_recall_num[{len(bm25Nodes)}]")
            for node in bm25Nodes:
                fmtText = truncate_text(node.node.text, 100).replace("\n", "//")
                logger.debug(f"score[{node.score:.2f}] content[{fmtText}]")

            for bm25Node in bm25Nodes:
                theNode = nodeIdToNode.get(bm25Node.node_id)
                if theNode is None:
                    nodes.append(bm25Node)
                    nodeIdToNode[bm25Node.node_id] = bm25Node
                    bm25Node.metadata["rrf"] = False 
                else:
                    theNode.metadata["rrf"] = True
                    theNode.score = HybridRetriever.__calcRRF(theNode.score, bm25Node.score)
        
        for node in nodes:
            if not node.metadata["rrf"]:
                node.score = HybridRetriever.__calcRRF(node.score, 0)

        nodes.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"final_recall[{len(nodes)}]")
        for node in nodes:
            fmtText = truncate_text(node.node.text, 100).replace("\n", "//")
            logger.debug(f"score[{node.score:.2f}] content[{fmtText}]")
        return nodes

    def __calcRRF(score0, score1):
        return 1.0/(60 + score0) + 1.0/(60 + score1)