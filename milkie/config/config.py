from abc import ABC
from dataclasses import dataclass
import json
from typing import List
from transformers import BitsAndBytesConfig

import torch

from milkie.prompt.prompt import Loader
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
    GEN_OPENAI = 1
    AZURE_OPENAI = 2

class FRAMEWORK(Enum):
    NONE = 0
    HUGGINGFACE = 1
    VLLM = 2
    LMDEPLOY = 3

class EmbeddingType(Enum):
    HUGGINGFACE = 0

class ChunkAugmentType(Enum):
    NONE = 0
    SIMPLE = 1

class RerankerType(Enum):
    NONE = 0
    FLAGEMBED = 1

class RerankPosition(Enum):
    NONE = 0
    SIMPLE = 1

class Device(Enum):
    CPU = 0
    CUDA = 1
    MPS = 2

    def getDevice(device):
        if device == Device.CPU.name:
            return "cpu"
        elif device == Device.CUDA.name:
            return "cuda"
        elif device == Device.MPS.name:
            return "mps"
        else:
            raise Exception("Device not supported")

class RewriteStrategy(Enum):
    NONE = 0
    QUERY_REWRITE = 1
    HYDE = 2

class QuantType(Enum):
    NONE = 0
    INT8 = 1
    INT4 = 2

class QuantMethod(Enum):
    NONE = 0
    GPTQ = 1
    AWQ = 2

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
        if not config:
            return None
        
        configs = []
        for singleConfig in config:
            memoryTermConfig = MemoryTermConfig.fromArgs(singleConfig)
            configs.append(memoryTermConfig)
        return MemoryConfig(memoryConfig=configs)

class LLMModelArgs(BaseConfig):
    def __init__(
            self,
            attnImplementation :str, 
            quantizationType :QuantType,
            torchCompile :bool):
        self.attnImplementation = attnImplementation
        self.quantizationType = quantizationType
        self.torchCompile = torchCompile
        self.torchDtype = torch.float16
        self.trustRemoteCode = True

    def fromArgs(config :dict):
        quantizationType = QuantType.NONE
        if config["quantization_type"] == QuantType.INT8.name:
            quantizationType = QuantType.INT8
        elif config["quantization_type"] == QuantType.INT4.name:
            quantizationType = QuantType.INT4
        
        llmModelArgs = LLMModelArgs(
            attnImplementation=config["attn_implementation"] if "attn_implementation" in config else None,
            quantizationType=quantizationType,
            torchCompile=config["torch_compile"] if "torch_compile" in config else False)
        return llmModelArgs

    def toJson(self):
        result = {
            "torch_dtype": self.torchDtype,
            "trust_remote_code": self.trustRemoteCode,
        }

        if self.attnImplementation:
            result["attn_implementation"] = self.attnImplementation

        quantizationConfig = None
        if self.quantizationType == QuantType.INT8:
            quantizationConfig = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantizationType == QuantType.INT4:
            quantizationConfig = BitsAndBytesConfig(load_in_4bit=True)
        
        if quantizationConfig:
            result["quantization_config"] = quantizationConfig

        if self.torchCompile:
            result["torch_compile"] = self.torchCompile
        return result

    def __repr__(self) -> str:
        result = {
            "torch_dtype": str(self.torchDtype),
            "trust_remote_code": self.trustRemoteCode,
        }

        if self.attnImplementation:
            result["attn_implementation"] = self.attnImplementation

        result["quantization_type"] = self.quantizationType.name
        if self.torchCompile:
            result["torch_compile"] = self.torchCompile
        return json.dumps(result)
        
class LLMGenerationArgs(BaseConfig):
    def __init__(
            self,
            repetitionPenalty: float,
            temperature :float,
            topK :int,
            topP :float,
            doSample :bool,
            useCache :bool,
            promptLookupNumTokens: int):
        self.repetitionPenalty = repetitionPenalty
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.doSample = doSample
        self.useCache = useCache
        self.promptLookupNumTokens = promptLookupNumTokens

    def fromArgs(config :dict):
        llmGenerationArgs = LLMGenerationArgs(
            repetitionPenalty=config["repetition_penalty"] if "repetition_penalty" in config else 1.0,
            temperature=config["temperature"] if "temperature" in config else 0,
            topK=config["top_k"] if "top_k" in config else -1,
            topP=config["top_p"] if "top_p" in config else 1.0,
            doSample=config["do_sample"] if "do_sample" in config else False,
            useCache=config["use_cache"] if "use_cache" in config else True,
            promptLookupNumTokens=config["prompt_lookup_num_tokens"] if "prompt_lookup_num_tokens" in config else None,
        )
        return llmGenerationArgs

    def toJson(self):
        result = {
            "repetition_penalty": self.repetitionPenalty,
            "temperature": self.temperature,
            "top_k" : self.topK,
            "top_p": self.topP,
            "do_sample": self.doSample,
            "use_cache": self.useCache,
        }

        if self.promptLookupNumTokens:
            result["prompt_lookup_num_tokens"] = self.promptLookupNumTokens
        return result

@dataclass
class LLMConfig(BaseConfig):
    def __init__(
            self, 
            type :LLMType, 
            model :str, 
            systemPrompt :str,
            ctxLen :int = 0,
            batchSize :int = 1,
            tensorParallelSize :int = 1,
            framework :FRAMEWORK = FRAMEWORK.NONE,
            device :int = None,
            port :int = None,
            deploymentName :str = None,
            apiKey :str = None,
            endpoint :str = None,
            apiVersion :str = None,
            modelArgs :LLMModelArgs = None,
            generationArgs :LLMGenerationArgs = None):
        self.type = type
        self.model = model
        self.systemPrompt = systemPrompt
        self.ctxLen = ctxLen
        self.batchSize = batchSize
        self.tensorParallelSize = tensorParallelSize
        self.framework = framework
        self.device = device
        self.port = port
        self.deploymentName = deploymentName
        self.apiKey = apiKey
        self.endpoint = endpoint
        self.apiVersion = apiVersion
        self.modelArgs = modelArgs
        self.generationArgs = generationArgs
    
    def fromArgs(config :dict):
        framework = FRAMEWORK.HUGGINGFACE
        if "framework" in config.keys():
            if config["framework"] == FRAMEWORK.VLLM.name:
                framework = FRAMEWORK.VLLM
            elif config["framework"] == FRAMEWORK.LMDEPLOY.name:
                framework = FRAMEWORK.LMDEPLOY

        device = None
        if "device" in config.keys():
            device = config["device"]
            device = Device.getDevice(device)

        port = None
        if "port" in config.keys():
            port = config["port"]
        
        modelArgs = None
        if "model_args" in config.keys():
            modelArgs = LLMModelArgs.fromArgs(config["model_args"])
        
        generationArgs = None
        if "generation_args" in config.keys():
            generationArgs = LLMGenerationArgs.fromArgs(config["generation_args"])

        systemPrompt = Loader.load(config["system_prompt"]) if "system_prompt" in config else None
        if config["type"] == LLMType.HUGGINGFACE.name:
            return LLMConfig(
                type=LLMType.HUGGINGFACE, 
                model=config["model"],
                systemPrompt=systemPrompt,
                ctxLen=config["ctx_len"],
                batchSize=config["batch_size"],
                tensorParallelSize=config["tensor_parallel_size"],
                framework=framework,
                device=device,
                port=port,
                modelArgs=modelArgs,
                generationArgs=generationArgs)
        elif config["type"] == LLMType.GEN_OPENAI.name:
            return LLMConfig(
                type=LLMType.GEN_OPENAI,
                model=config["model"],
                systemPrompt=systemPrompt,
                ctxLen=config["ctx_len"],
                apiKey=config["api_key"],
                endpoint=config["endpoint"],
                generationArgs=generationArgs,)
        elif config["type"] == LLMType.AZURE_OPENAI.name:
            return LLMConfig(
                type=LLMType.AZURE_OPENAI,
                model=config["model"],
                batchSize=config["batch_size"],
                deploymentName=config["deployment_name"],
                apiKey=config["api_key"],
                endpoint=config["endpoint"],
                apiVersion=config["api_version"],
                modelArgs=modelArgs,
                generationArgs=generationArgs)
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
        self.device = Device.getDevice(device)

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
            chunkOverlap=config["chunk_overlap"]) if config else None

class RerankConfig(BaseConfig):
    def __init__(
            self, 
            rerankerType :RerankerType, 
            model :str,
            rerankTopK :int,
            rerankPosition :RerankPosition):
        self.rerankerType = rerankerType
        self.model = model
        self.rerankTopK = rerankTopK
        self.rerankPosition = rerankPosition

class RetrievalConfig(BaseConfig):
    def __init__(
            self, 
            rewriteStrategy :RewriteStrategy,
            channelRecall :int,
            similarityTopK :int,
            blockSize :int,
            chunkAugmentType :ChunkAugmentType,
            rerankerConfig :RerankConfig):
        self.rewriteStrategy = rewriteStrategy
        self.channelRecall = channelRecall
        self.similarityTopK = similarityTopK 
        self.blockSize = blockSize
        self.chunkAugmentType = chunkAugmentType
        self.rerankerConfig = rerankerConfig

    def fromArgs(config :dict):
        chunkAugmentType = ChunkAugmentType.NONE
        if "chunk_augment" in config.keys() and config["chunk_augment"] == ChunkAugmentType.SIMPLE.name:
            chunkAugmentType = ChunkAugmentType.SIMPLE
        
        rerankerType = RerankerType.NONE
        if config["reranker"]["name"] == RerankerType.FLAGEMBED.name:
            rerankerType = RerankerType.FLAGEMBED
        
        rewriteStrategy = RewriteStrategy.NONE
        if config["rewrite_strategy"] == RewriteStrategy.HYDE.name:
            rewriteStrategy = RewriteStrategy.HYDE
        elif config["rewrite_strategy"] == RewriteStrategy.QUERY_REWRITE.name:
            rewriteStrategy = RewriteStrategy.QUERY_REWRITE

        positionConfig = RerankPosition.NONE
        if config["reranker"]["position"] == RerankPosition.SIMPLE.name:
            positionConfig = RerankPosition.SIMPLE
            
        reranker = RerankConfig(
            rerankerType=rerankerType,
            model=config["reranker"]["model"],
            rerankTopK=config["similarity_top_k"],
            rerankPosition=positionConfig)

        return RetrievalConfig(
            rewriteStrategy=rewriteStrategy,
            channelRecall=config["channel_recall"],
            similarityTopK=config["similarity_top_k"],
            blockSize=config["block_size"],
            chunkAugmentType=chunkAugmentType,
            rerankerConfig=reranker)

class AgentType(Enum):
    QA = 0
    PROMPT = 1

class SingleAgentConfig(BaseConfig):
    def __init__(
            self, 
            config :str, 
            type :AgentType,
            prompt :str):
        self.config = config
        self.type = type
        self.prompt = prompt

class PromptAgentConfig(SingleAgentConfig):
    def __init__(
            self, 
            config: str, 
            type: AgentType,
            prompt :str):
        super().__init__(config, type, prompt)

    def fromArgs(config :dict):
        return PromptAgentConfig(
            config=config["config"],
            prompt=config["prompt"] if "prompt" in config else None,
            type=AgentType.PROMPT)

class QAAgentConfig(SingleAgentConfig):
    def __init__(
            self, 
            config :str,
            type :AgentType,
            prompt :str,
            memoryConfig :MemoryConfig,
            indexConfig :IndexConfig,
            retrievalConfig :RetrievalConfig):
        super().__init__(config, type, prompt)
        self.memoryConfig = memoryConfig
        self.indexConfig = indexConfig
        self.retrievalConfig = retrievalConfig

    def fromArgs(config :dict):
        return QAAgentConfig(
            config=config["config"],
            type=AgentType.QA,
            prompt=config["prompt"] if "prompt" in config else None,
            memoryConfig=MemoryConfig.fromArgs(config["memory"]) if "memory" in config else None,
            indexConfig=IndexConfig.fromArgs(config["index"]) if "index" in config else None,
            retrievalConfig=RetrievalConfig.fromArgs(config["retrieval"]) if "retrieval" in config else None)

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

    def getConfig(self, config :str) -> SingleAgentConfig:
        return self.agentMap.get(config)

class EmailConfig(BaseConfig):
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    @classmethod
    def fromArgs(cls, config: dict):
        return cls(
            smtp_server=config["smtp_server"],
            smtp_port=config["smtp_port"],
            username=config["username"],
            password=config["password"]
        )

class ToolsConfig(BaseConfig):
    def __init__(self, email_config: EmailConfig):
        self.email_config = email_config

    def getEmailConfig(self) -> EmailConfig:
        return self.email_config

    @classmethod
    def fromArgs(cls, config: dict):
        email_config = EmailConfig.fromArgs(config.get("email", {}))
        return cls(email_config=email_config)

class GlobalConfig(BaseConfig):
    instanceCnt = 0
    
    def __init__(self, config):
        if GlobalConfig.instanceCnt == 0:
            GlobalConfig.instanceCnt = 1
        elif GlobalConfig.instanceCnt >= 0:
            raise Exception("GlobalConfig can only be initialized once")
        
        if type(config) == str:
            config = loadFromYaml(config)
        self.initFromDict(config)

    def initFromDict(self, config: dict):
        self.llmConfig = LLMConfig.fromArgs(config["llm"])
        self.llmCodeConfig = LLMConfig.fromArgs(config["llm_code"])
        self.embeddingConfig = EmbeddingConfig.fromArgs(config["embedding"]) if "embedding" in config.keys() else None
        self.agentsConfig: AgentsConfig = AgentsConfig.fromArgs(config["agents"])
        self.memoryConfig = MemoryConfig.fromArgs(config["memory"])
        self.indexConfig = IndexConfig.fromArgs(config["index"])
        self.retrievalConfig = RetrievalConfig.fromArgs(config["retrieval"]) if "retrieval" in config.keys() else None
        self.toolsConfig = ToolsConfig.fromArgs(config.get("tools", {}))

        for agentConfig in self.agentsConfig.agentConfigs:
            if agentConfig.type == AgentType.QA:
                if agentConfig.memoryConfig is None:
                    agentConfig.memoryConfig = self.memoryConfig

                if agentConfig.indexConfig is None:
                    agentConfig.indexConfig = self.indexConfig

                if agentConfig.retrievalConfig is None:
                    agentConfig.retrievalConfig = self.retrievalConfig

    def getLLMConfig(self) -> LLMConfig:
        return self.llmConfig

    def getLLMCodeConfig(self) -> LLMConfig:
        return self.llmCodeConfig
    
    def getEmbeddingConfig(self) -> EmbeddingConfig:
        return self.embeddingConfig
    
    def getAgentConfig(self, config :str) -> SingleAgentConfig:
        return self.agentsConfig.getConfig(config)

    def getMemoryConfig(self) -> MemoryConfig:
        return self.memoryConfig

    def getIndexConfig(self) -> IndexConfig:
        return self.indexConfig

    def getRetrievalConfig(self) -> RetrievalConfig:
        return self.retrievalConfig

    def getToolsConfig(self) -> ToolsConfig:
        return self.toolsConfig
    
    def getEmailConfig(self) -> EmailConfig:
        return self.toolsConfig.getEmailConfig()