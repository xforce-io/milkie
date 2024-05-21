from typing import List
import jieba

from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.response_synthesizers.factory import get_response_synthesizer
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.response_synthesizers.type import ResponseMode
from llama_index.legacy.schema import NodeWithScore

from milkie.agent.prompt_agent import PromptAgent
from milkie.config.config import GlobalConfig, RerankPosition, RetrievalConfig, RewriteStrategy
from milkie.context import Context
from milkie.custom_refine_program import CustomProgramFactory
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.prompt.test_prompts import candidateTextQAPromptSel, candidateRefinePromptSel, candidateTextQAPromptImpl, candidateRefinePromptImpl
from milkie.retrieval.position_reranker import PositionReranker
from milkie.retrieval.reranker import Reranker
from milkie.retrieval.retrievers import HybridRetriever

def chineseTokenizer(text) :
    return list(jieba.cut(text, cut_all=False))

class RetrievalModule:
    def __init__(
            self, 
            globalConfig :GlobalConfig,
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
            similarity_top_k=self.retrievalConfig.channelRecall,
            tokenizer=chineseTokenizer,)

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
            program_factory=CustomProgramFactory(memoryWithIndex.settings.llm),
            structured_answer_filtering=True,
            text_qa_template=candidateTextQAPromptSel(
                globalConfig.getLLMConfig().systemPrompt,
                "qa_init"),
            refine_template=candidateRefinePromptSel("qa_refine"),
        )

        self.engine = RetrieverQueryEngine.from_args(
            retriever=self.hybridRetriever,
            node_postprocessors=nodePostProcessors,
            service_context=memoryWithIndex.serviceContext,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=candidateTextQAPromptImpl("qa_init"),
            refine_template=candidateRefinePromptImpl("qa_refine"),
            response_synthesizer=responseSynthesizer)

    def retrieve(self, context :Context) -> List[NodeWithScore]:
        curQuery = context.getCurQuery()
        if self.rewriteAgent:
            self.rewriteAgent.setContext(context)
            rewriteResp = self.rewriteAgent.taskBatch(
                query=None,
                argsList=[{"query_str":curQuery}])
            curQuery = rewriteResp[0].response

        result = self.engine.retrieve(QueryBundle(curQuery))
        context.setRetrievalResult(result)
        return result