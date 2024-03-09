from typing import List
from llama_index import QueryBundle, get_response_synthesizer
from milkie.agent.prompt_agent import PromptAgent
from milkie.config.config import RerankPosition, RetrievalConfig, RewriteStrategy
from milkie.context import Context
from milkie.custom_refine_program import CustomProgramFactory
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.prompt.test_prompts import CANDIDATE_REFINE_PROMPT_SEL, CANDIDATE_TEXT_QA_PROMPT_IMPL, CANDIDATE_REFINE_PROMPT_IMPL, CANDIDATE_TEXT_QA_PROMPT_SEL
from milkie.retrieval.position_reranker import PositionReranker
from milkie.retrieval.reranker import Reranker
from milkie.retrieval.retrievers import HybridRetriever

from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.schema import NodeWithScore


class RetrievalModule:
    def __init__(
            self, 
            retrievalConfig :RetrievalConfig,
            memoryWithIndex :MemoryWithIndex):
        self.rewriteAgent = None
        if retrievalConfig.rewriteStrategy == RewriteStrategy.HYDE:
            self.rewriteAgent = PromptAgent(
                context=None, 
                config="hyde")
        elif retrievalConfig.rewriteStrategy == RewriteStrategy.QUERY_REWRITE:
            self.rewriteAgent = PromptAgent(
                context=None, 
                config="query_rewrite")
        
        self.retrievalConfig = retrievalConfig

        self.denseRetriever = memoryWithIndex.index.denseIndex.as_retriever(
            similarity_top_k=self.retrievalConfig.channelRecall)

        self.sparseRetriever = BM25Retriever.from_defaults(
            docstore=memoryWithIndex.index.denseIndex.docstore,
            similarity_top_k=self.retrievalConfig.channelRecall)

        self.hybridRetriever = HybridRetriever(
            self.denseRetriever, 
            self.sparseRetriever)

        nodePostProcessors = []
        reranker = Reranker(self.retrievalConfig.rerankerConfig) 
        if reranker.reranker:
            nodePostProcessors.append(reranker.reranker)

        if self.retrievalConfig.rerankerConfig.rerankPosition == RerankPosition.SIMPLE:
            positionReranker = PositionReranker()
            nodePostProcessors.append(positionReranker)

        responseSynthesizer = get_response_synthesizer(
            service_context=memoryWithIndex.serviceContext,
            program_factory=CustomProgramFactory(memoryWithIndex.serviceContext.llm),
            structured_answer_filtering=True,
            text_qa_template=CANDIDATE_TEXT_QA_PROMPT_SEL,
            refine_template=CANDIDATE_REFINE_PROMPT_SEL,
        )

        self.engine = RetrieverQueryEngine.from_args(
            retriever=self.hybridRetriever,
            node_postprocessors=nodePostProcessors,
            service_context=memoryWithIndex.serviceContext,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=CANDIDATE_TEXT_QA_PROMPT_IMPL,
            refine_template=CANDIDATE_REFINE_PROMPT_IMPL,
            response_synthesizer=responseSynthesizer)

    def retrieve(self, context :Context) -> List[NodeWithScore]:
        curQuery = context.getCurQuery()
        if self.rewriteAgent:
            self.rewriteAgent.setContext(context)
            rewriteResp = self.rewriteAgent.task(
                curQuery,
                query_str=curQuery,)
            curQuery = rewriteResp.response

        result = self.engine.retrieve(QueryBundle(curQuery))
        context.setRetrievalResult(result)
        return result