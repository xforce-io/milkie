from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import List

from milkie.utils.data_utils import loadFromYaml


@dataclass(frozen=True)
class BaseConfig(ABC):
    pass

from enum import Enum

class MemoryType(Enum):
    LONG_TERM = 0

class LongTermMemorySource(Enum):
    LOCAL = 0

class LLMType(Enum):
    HUGGINGFACE = 0
    AZURE_OPENAI = 1

class EmbeddingType(Enum):
    HUGGINGFACE = 0

class RerankerType(Enum):
    FLAGEMBED = 0

class MemoryTermConfig(BaseConfig):
    def __init__(
            self, 
            type :MemoryType, 
            source :LongTermMemorySource,
            path :str):
        self.type = type 
        self.source = source
        self.path = path

class LLMConfig(BaseConfig):
    def __init__(
            self, 
            type :LLMType, 
            model :str, 
            ctxLen :int = 0,
            deploymentName :str = None,
            apiKey :str = None,
            azureEndpoint :str = None,
            apiVersion :str = None):
        self.type = type
        self.model = model
        self.ctxLen = ctxLen
        self.deploymentName = deploymentName
        self.apiKey = apiKey
        self.azureEndpoint = azureEndpoint
        self.apiVersion = apiVersion

class EmbeddingConfig(BaseConfig):
    def __init__(self, type :EmbeddingType, model :str):
        self.type = type
        self.model = model 

class IndexConfig(BaseConfig):
    def __init__(self, chunkSize :int):
        self.chunkSize = chunkSize

class RerankConfig(BaseConfig):
    def __init__(self, rerankerType :RerankerType, rerankTopK :int):
        self.rerankerType = rerankerType
        self.rerankTopK = rerankTopK

class RetrievalConfig(BaseConfig):
    def __init__(
            self, 
            channelRecall:int,
            similarityTopK :int,
            rerankerConfig :RerankConfig):
        self.channelRecall = channelRecall
        self.similarityTopK = similarityTopK 
        if rerankerConfig["name"] == RerankerType(0).name:
            self.reranker = RerankConfig(
                rerankerConfig["name"],
                self.similarityTopK)
        else:
            self.reranker = None

class GlobalConfig(BaseConfig):
    def __init__(self, configPath :str):
        config = loadFromYaml(configPath)
        self.__init__(config)

    def __init__(self, config :dict):
        self.memoryConfig = self.__buildMemoryConfig(config["memory"])
        self.llmConfig = self.__buildLLMConfig(config["llm"])
        self.embeddingConfig = self.__buildEmbeddingConfig(config["embedding"])
        self.indexConfig = self.__buildIndexConfig(config["index"])
        self.retrievalConfig = self.__buildRetrievalConfig(config["retrieval"])

    def __buildMemoryConfig(self, memoryConfig):
        configs = []
        for singleConfig in memoryConfig:
            memoryTermConfig = self.__buildMemoryTermConfig(singleConfig)
            configs.append(memoryTermConfig)
        return configs

    def __buildMemoryTermConfig(self, memoryTermConfig):
        if memoryTermConfig["type"] == MemoryType.LONG_TERM.name:
            if memoryTermConfig["source"] == LongTermMemorySource.LOCAL.name:
                return MemoryTermConfig(
                    memoryTermConfig["type"], 
                    memoryTermConfig["source"],
                    memoryTermConfig["path"])
            else:
                raise Exception(f"Long term memory source not supported[{memoryTermConfig['source']}]")
        else:
            raise Exception(f"Memory type not supported[{memoryTermConfig['type']}]")

    def __buildLLMConfig(self, configLLM):
        if configLLM["type"] == LLMType.HUGGINGFACE.name:
            return LLMConfig(
                LLMType.HUGGINGFACE, 
                model=configLLM["model"],
                ctxLen=configLLM["ctx_len"])
        elif configLLM["type"] == LLMType.AZURE_OPENAI.name:
            return LLMConfig(
                LLMType.AZURE_OPENAI,
                model=configLLM["model"],
                deploymentName=configLLM["deployment_name"],
                apiKey=configLLM["api_key"],
                azureEndpoint=configLLM["azure_endpoint"],
                apiVersion=configLLM["api_version"])
        else:
            raise Exception("LLM type not supported")
    
    def __buildEmbeddingConfig(self, configEmbedding):
        if configEmbedding["type"] == EmbeddingType(0).name:
            return EmbeddingConfig(EmbeddingType(0), configEmbedding["model"])
        else:
            raise Exception("Embedding type not supported")

    def __buildIndexConfig(self, configIndex):
        return IndexConfig(configIndex["chunk_size"])

    def __buildRetrievalConfig(self, configRetrieval):
        return RetrievalConfig(
            configRetrieval["channel_recall"],
            configRetrieval["similarity_top_k"],
            configRetrieval["reranker"])