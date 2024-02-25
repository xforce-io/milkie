from llama_index import QueryBundle, get_response_synthesizer
from milkie.config.config import QAAgentConfig
from milkie.context import Context
from milkie.custom_refine_program import CustomRefineProgram
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.prompt.test_prompts import CANDIDATE_TEXT_QA_PROMPT_IMPL, CANDIDATE_REFINE_PROMPT_IMPL
from milkie.retrieval.position_reranker import PositionReranker
from milkie.retrieval.reranker import Reranker
from milkie.retrieval.retrievers import HybridRetriever

from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers.type import ResponseMode

class RetrievalModule:
    def __init__(
            self, 
            qaAgentConfig :QAAgentConfig,
            memoryWithIndex :MemoryWithIndex):
        self.retrievalConfig = qaAgentConfig.retrievalConfig

        self.denseRetriever = memoryWithIndex.index.denseIndex.as_retriever(
            similarity_top_k=self.retrievalConfig.channelRecall)

        self.sparseRetriever = BM25Retriever.from_defaults(
            docstore=memoryWithIndex.index.denseIndex.docstore,
            similarity_top_k=self.retrievalConfig.channelRecall)

        self.hybridRetriever = HybridRetriever(
            self.denseRetriever, 
            self.sparseRetriever)

        nodePostProcessors = []
        if self.retrievalConfig.reranker is not None:   
            reranker = Reranker(self.retrievalConfig.reranker) 
            nodePostProcessors.append(reranker.reranker)

            positionReranker = PositionReranker()
            nodePostProcessors.append(positionReranker)

        responseSynthesizer = get_response_synthesizer(
            service_context=memoryWithIndex.serviceContext,
            program_factory=CustomRefineProgram(),
        )

        self.engine = RetrieverQueryEngine.from_args(
            retriever=self.hybridRetriever,
            node_postprocessors=nodePostProcessors,
            service_context=memoryWithIndex.serviceContext,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=CANDIDATE_TEXT_QA_PROMPT_IMPL,
            refine_template=CANDIDATE_REFINE_PROMPT_IMPL,
            response_synthesizer=responseSynthesizer)

    def retrieve(self, context :Context):
        result = self.engine.retrieve(QueryBundle(context.getCurQuery()))
        context.setRetrievalResult(result)