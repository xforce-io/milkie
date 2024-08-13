from typing import List
import re
import jieba

from llama_index.core.schema import QueryBundle
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25.base import BM25Retriever

from milkie.agent.prompt_agent import PromptAgent
from milkie.agent.query_structure import QueryType
from milkie.config.config import ChunkAugmentType, GlobalConfig, RerankPosition, RetrievalConfig, RewriteStrategy
from milkie.context import Context
from milkie.custom_refine_program import CustomProgramFactory
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.prompt.test_prompts import candidateTextQAPromptSel, candidateRefinePromptSel, candidateTextQAPromptImpl, candidateRefinePromptImpl
from milkie.retrieval.chunk_augment import ChunkAugment
from milkie.retrieval.position_reranker import PositionReranker
from milkie.retrieval.reranker import Reranker
from milkie.retrieval.retrievers import HybridRetriever

def chineseTokenizer(text) :
    return list(jieba.cut(text, cut_all=True))

class RetrievalModule:
    def __init__(
            self, 
            globalConfig :GlobalConfig,
            retrievalConfig :RetrievalConfig,
            memoryWithIndex :MemoryWithIndex,
            context :Context):
        self.globalConfig = globalConfig
        self.retrievalConfig = retrievalConfig
        self.memoryWithIndex = memoryWithIndex
        self.context = context
            
        self.rewriteAgent = None
        if retrievalConfig.rewriteStrategy == RewriteStrategy.HYDE:
            self.rewriteAgent = PromptAgent(
                context=self.context, 
                config="hyde")
        elif retrievalConfig.rewriteStrategy == RewriteStrategy.QUERY_REWRITE:
            self.rewriteAgent = PromptAgent(
                context=self.context, 
                config="query_rewrite")

        self.denseRetriever = memoryWithIndex.index.denseIndex.as_retriever(
            similarity_top_k=self.retrievalConfig.channelRecall)

        self.sparseRetriever = BM25Retriever.from_defaults(
            docstore=memoryWithIndex.memory.storageContext.docstore,
            similarity_top_k=self.retrievalConfig.channelRecall,
            tokenizer=chineseTokenizer,)

        self.hybridRetriever = HybridRetriever(
            self.denseRetriever, 
            self.sparseRetriever,
            self.retrievalConfig.similarityTopK)

        self.nodePostProcessors = []

        reranker = Reranker(self.retrievalConfig.rerankerConfig) 
        if reranker.reranker:
            self.nodePostProcessors.append(reranker.reranker)

        if self.retrievalConfig.rerankerConfig.rerankPosition == RerankPosition.SIMPLE:
            positionReranker = PositionReranker()
            self.nodePostProcessors.append(positionReranker)

        self.chunkAugment = None
        if self.retrievalConfig.chunkAugmentType == ChunkAugmentType.SIMPLE:
            self.chunkAugment = ChunkAugment()
            self.nodePostProcessors.append(self.chunkAugment)
        
    def retrieve(self, context :Context, **kwargs) -> List[NodeWithScore]:
        if self.chunkAugment:
            self.chunkAugment.set_context(context)

        if context.getCurQuery().queryType == QueryType.FILEPATH:
            filepath = context.getCurQuery().query
            content = ""
            if filepath.endswith(".txt"):
                content = self._getTxtFileContent(filepath)
            elif filepath.endswith(".pdf"):
                content = self._getPdfFileContent(filepath)
            else:
                raise Exception(f"Unsupported file type[{filepath}]")
            
            if content is None:
                return None

            nodes = list()
            for i in range(0, len(content), self.retrievalConfig.blockSize):
                text = content[i:i+self.retrievalConfig.blockSize]
                textNode = TextNode(text=text)
                node = NodeWithScore(node=textNode)
                node.score = 1
                nodes.append(node)
            context.setRetrievalResult(nodes)
            return content

        responseSynthesizer = get_response_synthesizer(
            service_context=self.memoryWithIndex.serviceContext,
            program_factory=CustomProgramFactory(
                self.memoryWithIndex.settings.llm, 
                **kwargs),
            structured_answer_filtering=True,
            text_qa_template=candidateTextQAPromptSel(
                self.globalConfig.getLLMConfig().systemPrompt,
                "qa_init"),
            refine_template=candidateRefinePromptSel("qa_refine"),
        )

        self.engine = RetrieverQueryEngine.from_args(
            retriever=self.hybridRetriever,
            node_postprocessors=self.nodePostProcessors,
            service_context=self.memoryWithIndex.serviceContext,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=candidateTextQAPromptImpl("qa_init"),
            refine_template=candidateRefinePromptImpl("qa_refine"),
            response_synthesizer=responseSynthesizer)
        
        curQuery = context.getCurQuery().query
        if self.rewriteAgent:
            self.rewriteAgent.setContext(context)
            rewriteResp = self.rewriteAgent.executeBatch(
                query=None,
                argsList=[{"query_str":curQuery}],
                **kwargs)
            curQuery = curQuery + "|" + rewriteResp[0].response

        result = self.engine.retrieve(QueryBundle(curQuery))
        context.setRetrievalResult(result)
        return result

    @staticmethod
    def _getTxtFileContent(filePath :str) -> str:
        with open(filePath, "r") as f:
            return f.read()

    PatternChinese = re.compile(r'[\u4e00-\u9fff]')

    @staticmethod
    def _getPdfFileContent(filePath :str) -> str:
        import PyPDF2
        import re
        
        content = ""
        with open(filePath, "rb") as f:
            pdfReader = PyPDF2.PdfReader(f)
            content = "".join([page.extract_text() for page in pdfReader.pages])
            cleaned = re.sub(r"/G[A-Z0-9]+", "", content)
            cleaned = re.sub(r"\s+", "", cleaned)
        
        if len(cleaned) < 100:
            return None
        
        if bool(RetrievalModule.PatternChinese.search(cleaned)) :
            return cleaned
        return None