from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

from milkie.config.config import IndexConfig, MemoryConfig
from milkie.index.index import Index
from milkie.memory.memory import Memory
from milkie.settings import Settings

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
                llm=settings.llm)

        self.memory = Memory(
            memoryTermConfigs=memoryConfig.memoryConfig, 
            serviceContext=self.serviceContext)

        denseIndex = VectorStoreIndex(
            self.memory.nodes,
            storage_context=self.memory.storageContext,
            service_context=self.serviceContext)
        
        self.index = Index(denseIndex)