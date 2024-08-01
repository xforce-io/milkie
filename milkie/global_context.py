from typing import List, Optional, Sequence

from logging import Logger

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.sentence import (
    DEFAULT_CHUNK_SIZE,
    SENTENCE_CHUNK_OVERLAP,
    SentenceSplitter,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import DEFAULT_PADDING, PromptHelper

from milkie.config.config import GlobalConfig, LLMConfig
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.settings import Settings
from milkie.model_factory import ModelFactory

def getNodeParser(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
    callback_manager: Optional[CallbackManager] = None,
) -> NodeParser:
    """Get default node parser."""
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager or CallbackManager(),
        separator="\n\n",
    )

class CustomizedPromptHelper(PromptHelper):

    def __init__(self, llmConfig :LLMConfig):
        super().__init__(
            context_window=llmConfig.ctxLen,
            chunk_size_limit=llmConfig.ctxLen*3/4)
    
    def repack(
            self, 
            prompt: BasePromptTemplate, 
            text_chunks: Sequence[str], 
            padding: int = DEFAULT_PADDING,
            llm: Optional[LLM] = None) -> List[str]:
        chunks = super().repack(prompt, text_chunks, padding, llm)
        return chunks[:2]

class GlobalContext():
    
    def __init__(
            self, 
            globalConfig :GlobalConfig, 
            modelFactory :ModelFactory,
            logger :Logger):
        self.globalConfig = globalConfig
        self.modelFactory = modelFactory
        self.logger = logger
        self.settings = Settings(globalConfig, modelFactory)
        promptHelper = CustomizedPromptHelper(
            llmConfig=globalConfig.getLLMConfig()
        )

        self.serviceContext = ServiceContext.from_defaults(
            embed_model=self.settings.embedding,
            chunk_size=globalConfig.indexConfig.chunkSize if globalConfig.indexConfig else None,
            chunk_overlap=globalConfig.indexConfig.chunkOverlap if globalConfig.indexConfig else None,
            prompt_helper=promptHelper,
            llm=self.settings.llm.getLLM(),
            node_parser=getNodeParser())

        if globalConfig.memoryConfig and globalConfig.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                settings=self.settings,
                memoryConfig=self.globalConfig.memoryConfig,
                indexConfig=self.globalConfig.indexConfig,
                serviceContext=self.serviceContext)
        else:
            self.memoryWithIndex = None