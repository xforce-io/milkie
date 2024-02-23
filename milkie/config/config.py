from abc import ABC
from dataclasses import dataclass
from typing import List

from milkie.utils.data_utils import loadFromYaml


@dataclass
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

class Device(Enum):
    CPU = 0
    CUDA = 1
    MPS = 2

class MemoryTermConfig(BaseConfig):
    def __init__(
            self, 
            type :MemoryType, 
            source :LongTermMemorySource,
            path :str):
        self.type = type 
        self.source = source
        self.path = path
    
    def fromArgs(config :dict):
        if config["type"] == MemoryType.LONG_TERM.name:
            if config["source"] == LongTermMemorySource.LOCAL.name:
                return MemoryTermConfig(
                    type=config["type"],
                    source=config["source"],
                    path=config["path"])
            else:
                raise Exception(f"Long term memory source not supported[{config['source']}]")
        else:
            raise Exception(f"Memory type not supported[{config['type']}]")

class MemoryConfig(BaseConfig):
    def __init__(self, memoryConfig :List[MemoryTermConfig]):
        self.memoryConfig = memoryConfig

    def fromArgs(config :dict):
        configs = []
        for singleConfig in config:
            memoryTermConfig = MemoryTermConfig.fromArgs(singleConfig)
            configs.append(memoryTermConfig)
        return MemoryConfig(memoryConfig=configs)

@dataclass
class LLMConfig(BaseConfig):
    def __init__(
            self, 
            type :LLMType, 
            model :str, 
            ctxLen :int = 0,
            temperature :float = 0,
            deploymentName :str = None,
            apiKey :str = None,
            azureEndpoint :str = None,
            apiVersion :str = None):
        self.type = type
        self.model = model
        self.ctxLen = ctxLen
        self.temperature = temperature
        self.deploymentName = deploymentName
        self.apiKey = apiKey
        self.azureEndpoint = azureEndpoint
        self.apiVersion = apiVersion
    
    def fromArgs(config :dict):
        if config["type"] == LLMType.HUGGINGFACE.name:
            return LLMConfig(
                type=LLMType.HUGGINGFACE, 
                model=config["model"],
                ctxLen=config["ctx_len"],
                temperature=config["temperature"])
        elif config["type"] == LLMType.AZURE_OPENAI.name:
            return LLMConfig(
                type=LLMType.AZURE_OPENAI,
                model=config["model"],
                temperature=config["temperature"],
                deploymentName=config["deployment_name"],
                apiKey=config["api_key"],
                azureEndpoint=config["azure_endpoint"],
                apiVersion=config["api_version"])
        else:
            raise Exception("LLM type not supported")

class EmbeddingConfig(BaseConfig):
    def __init__(
            self, 
            type :EmbeddingType, 
            model :str,
            device :Device):
        self.type = type
        self.model = model 
        if device == Device.CPU.name:
            self.device = "cpu"
        elif device == Device.MPS.name:
            self.device = "mps"
        else:
            self.device = "cuda"

    def fromArgs(config :dict):
        if config["type"] == EmbeddingType.HUGGINGFACE.name:
            return EmbeddingConfig(
                type=EmbeddingType.HUGGINGFACE,
                model=config["model"],
                device=config["device"])
        else:
            raise Exception("Embedding type not supported")

class IndexConfig(BaseConfig):
    def __init__(
            self, 
            chunkSize :int,
            chunkOverlap :int):
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap

    def fromArgs(config :dict):
        return IndexConfig(
            chunkSize=config["chunk_size"],
            chunkOverlap=config["chunk_overlap"])

class RerankConfig(BaseConfig):
    def __init__(
            self, 
            rerankerType :RerankerType, 
            model :str,
            rerankTopK :int):
        self.rerankerType = rerankerType
        self.model = model
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
                rerankerType=rerankerConfig["name"],
                model=rerankerConfig["model"],
                rerankTopK=self.similarityTopK)
        else:
            self.reranker = None

    def fromArgs(config :dict):
        return RetrievalConfig(
            channelRecall=config["channel_recall"],
            similarityTopK=config["similarity_top_k"],
            rerankerConfig=config["reranker"])

class AgentType(Enum):
    QA = 0
    PROMPT = 1

class SingleAgentConfig(BaseConfig):
    def __init__(
            self, 
            config :str, 
            type :AgentType):
        self.config = config
        self.type = type

class PromptAgentConfig(SingleAgentConfig):
    def __init__(
            self, 
            config: str, 
            type: AgentType):
        super().__init__(config, type)

    def fromArgs(config :dict):
        return PromptAgentConfig(
            config=config["config"],
            type=AgentType.PROMPT)

class QAAgentConfig(SingleAgentConfig):
    def __init__(
            self, 
            config :str,
            type :AgentType,
            memoryConfig :MemoryConfig,
            indexConfig :IndexConfig,
            retrievalConfig :RetrievalConfig):
        super().__init__(config, type)
        self.memoryConfig = memoryConfig
        self.indexConfig = indexConfig
        self.retrievalConfig = retrievalConfig

    def fromArgs(config :dict):
        return QAAgentConfig(
            config=config["config"],
            type=AgentType.QA,
            memoryConfig=MemoryConfig.fromArgs(config["memory"]),
            indexConfig=IndexConfig.fromArgs(config["index"]),
            retrievalConfig=RetrievalConfig.fromArgs(config["retrieval"]))

def createAgentConfig(config :dict):
    if config["type"] == AgentType.QA.name:
        return QAAgentConfig.fromArgs(config)
    elif config["type"] == AgentType.PROMPT.name:
        return PromptAgentConfig.fromArgs(config)
    else:
        raise Exception("Agent type not supported")

class AgentsConfig(BaseConfig):
    def __init__(self, agentConfigs :List[SingleAgentConfig]):
        self.agentConfigs = agentConfigs
        self.agentMap = {}
        for agentConfig in agentConfigs:
            self.agentMap[agentConfig.config] = agentConfig
    
    def fromArgs(config :dict):
        configs = []
        for singleConfig in config:
            agentConfig = createAgentConfig(singleConfig)
            configs.append(agentConfig)
        return AgentsConfig(configs)

    def getConfig(self, config :str):
        return self.agentMap.get(config)

class GlobalConfig(BaseConfig):
    def __init__(self, configPath :str):
        config = loadFromYaml(configPath)
        self.__init__(config)

    def __init__(self, config :dict):
        self.llmConfig = LLMConfig.fromArgs(config["llm"])
        self.embeddingConfig = EmbeddingConfig.fromArgs(config["embedding"])
        self.agentsConfig :AgentsConfig = AgentsConfig.fromArgs(config["agents"])

    def getLLMConfig(self):
        return self.llmConfig
    
    def getEmbeddingConfig(self):
        return self.embeddingConfig
    
    def getAgentConfig(self, config :str):
        return self.agentsConfig.getConfig(config)