from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

from milkie.config.config import IndexConfig, LongTermMemorySource, MemoryConfig, MemoryTermConfig, MemoryType
from milkie.index.index import Index
from milkie.ontology.memory.memory import Memory
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
        self.serviceContext = serviceContext
        self.index = None
        self.memory = None

    def rebuildFromLocalDir(self, localDir :str):
        self.memoryConfig = MemoryConfig([
            MemoryTermConfig(
                type=MemoryType.LONG_TERM,
                source=LongTermMemorySource.LOCAL,
                path=localDir)
        ])
        self.memory = None

    def getIndex(self):
        self._lazyBuildIndex()
        return self.index

    def getMemory(self):
        self._lazyBuildIndex()
        return self.memory

    def _lazyBuildIndex(self):
        if self.memory is None:
            self.memory = Memory(
                memoryTermConfigs=self.memoryConfig.memoryConfig, 
                serviceContext=self.serviceContext)

            denseIndex = VectorStoreIndex(
                self.memory.nodes,
                storage_context=self.memory.storageContext,
                service_context=self.serviceContext)
            
            self.index = Index(denseIndex)