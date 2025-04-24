from typing import List, Optional, Sequence

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.sentence import (
    SentenceSplitter,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import DEFAULT_PADDING, PromptHelper

from milkie.model_factory import ModelFactory
from milkie.vm.vm import VMFactory

def getNodeParser(
    chunk_size: int,
    chunk_overlap: int,
    callback_manager: Optional[CallbackManager] = None,
) -> NodeParser:
    """Get default node parser."""
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager or CallbackManager(),
        paragraph_separator="\n",
    )

from milkie.config.config import LLMBasicConfig

class CustomizedPromptHelper(PromptHelper):

    def __init__(
            self, 
            llmBasicConfig :LLMBasicConfig):
        ctxLen = llmBasicConfig.ctxLen
        super().__init__(
            context_window=ctxLen,
            chunk_size_limit=ctxLen*3/4)
    
    def repack(
            self, 
            prompt: BasePromptTemplate, 
            text_chunks: Sequence[str], 
            padding: int = DEFAULT_PADDING,
            llm: Optional[LLM] = None) -> List[str]:
        chunks = super().repack(prompt, text_chunks, padding, llm)
        return chunks[:2]

from milkie.config.config import GlobalConfig
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.settings import Settings

class GlobalContext():
    
    def __init__(
            self, 
            globalConfig :GlobalConfig, 
            modelFactory :ModelFactory):
        self.globalConfig = globalConfig
        self.modelFactory = modelFactory
        self.settings = Settings(globalConfig, modelFactory)
        promptHelper = CustomizedPromptHelper(
            llmBasicConfig=self.settings.llmBasicConfig)

        self.serviceContext = ServiceContext.from_defaults(
            embed_model=self.settings.embedding,
            chunk_size=globalConfig.indexConfig.chunkSize,
            chunk_overlap=globalConfig.indexConfig.chunkOverlap,
            prompt_helper=promptHelper,
            llm=self.settings.llmDefault.getLLM(),
            node_parser=getNodeParser(
                chunk_size=globalConfig.indexConfig.chunkSize,
                chunk_overlap=globalConfig.indexConfig.chunkOverlap,
            ))

        if globalConfig.memoryConfig and globalConfig.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                settings=self.settings,
                memoryConfig=self.globalConfig.memoryConfig,
                indexConfig=self.globalConfig.indexConfig,
                serviceContext=self.serviceContext)
        else:
            self.memoryWithIndex = None

        self.env = None
        self.vm = VMFactory.createVM(self.globalConfig.vmConfig)

    def setEnv(self, env):
        self.env = env
        
    def getEnv(self):
        return self.env

    @staticmethod
    def create(configPath :str = None):
        configPath = configPath if configPath else "config/global.yaml"

        globalConfig = GlobalConfig(configPath)
        modelFactory = ModelFactory()
        return GlobalContext(globalConfig, modelFactory)