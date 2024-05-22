from typing import Optional

from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.sentence import (
    DEFAULT_CHUNK_SIZE,
    SENTENCE_CHUNK_OVERLAP,
    SentenceSplitter,
)
from llama_index.core.callbacks.base import CallbackManager


from milkie.config.config import IndexConfig, MemoryConfig
from milkie.index.index import Index
from milkie.memory.memory import Memory
from milkie.settings import Settings

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

class MemoryWithIndex():

    def __init__(
            self,
            settings :Settings,
            memoryConfig :MemoryConfig,
            indexConfig :IndexConfig,
            serviceContext :ServiceContext):
        self.settings = settings
        self.memoryConfig = memoryConfig
        self.indexConfig = indexConfig

        if serviceContext:
            self.serviceContext = serviceContext
        else:
            self.serviceContext = ServiceContext.from_defaults(
                embed_model=settings.embedding,
                chunk_size=indexConfig.chunkSize,
                chunk_overlap=indexConfig.chunkOverlap,
                llm=settings.llm,
                node_parser=getNodeParser)

        self.memory = Memory(
            memoryTermConfigs=memoryConfig.memoryConfig, 
            serviceContext=self.serviceContext)

        denseIndex = VectorStoreIndex(
            self.memory.nodes,
            storage_context=self.memory.storageContext,
            service_context=self.serviceContext)
        
        self.index = Index(denseIndex)