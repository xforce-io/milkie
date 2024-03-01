from llama_index import ServiceContext
from milkie.config.config import GlobalConfig
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.settings import Settings
from milkie.model_factory import ModelFactory


class GlobalContext():
    
    def __init__(
            self, 
            globalConfig :GlobalConfig, 
            modelFactory :ModelFactory):
        self.globalConfig = globalConfig
        self.modelFactory = modelFactory
        self.settings = Settings(globalConfig, modelFactory)
        self.serviceContext = ServiceContext.from_defaults(
            embed_model=self.settings.embedding,
            chunk_size=globalConfig.indexConfig.chunkSize if globalConfig.indexConfig else None,
            chunk_overlap=globalConfig.indexConfig.chunkOverlap if globalConfig.indexConfig else None,
            llm=self.settings.llm)

        if globalConfig.memoryConfig and globalConfig.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                settings=self.settings,
                memoryConfig=self.globalConfig.memoryConfig,
                indexConfig=self.globalConfig.indexConfig,
                serviceContext=self.serviceContext)
        else:
            self.memoryWithIndex = None