from llama_index import QueryBundle
from milkie.context import Context
from milkie.retrieval.position_reranker import PositionReranker
from milkie.retrieval.reranker import Reranker
from milkie.retrieval.retrievers import HybridRetriever

from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers.type import ResponseMode

class RetrievalModule:
    def __init__(self, context :Context):
        self.context = context
        self.retrievalConfig = context.config.retrievalConfig

        self.denseRetriever = context.index.denseIndex.as_retriever(
            similarity_top_k=self.retrievalConfig.similarityTopK)

        self.sparseRetriever = BM25Retriever.from_defaults(
            docstore=context.index.denseIndex.docstore,
            similarity_top_k=self.retrievalConfig.similarityTopK
        )

        self.hybridRetriever = HybridRetriever(
            self.denseRetriever, 
            self.sparseRetriever)

        nodePostProcessors = []
        if self.retrievalConfig.reranker is not None:   
            reranker = Reranker(self.retrievalConfig.reranker) 
            nodePostProcessors.append(reranker.reranker)

            positionReranker = PositionReranker()
            nodePostProcessors.append(positionReranker)

        context.engine = RetrieverQueryEngine.from_args(
            retriever=self.hybridRetriever,
            node_postprocessors=nodePostProcessors,
            service_context=context.serviceContext,
            response_mode=ResponseMode.COMPACT)

    def retrieve(self, context :Context):
        result = context.engine.retrieve(QueryBundle(context.getCurQuery()))
        context.setRetrievalResult(result)