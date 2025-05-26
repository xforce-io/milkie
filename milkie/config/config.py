from abc import ABC
from dataclasses import dataclass
import json
from typing import List
import os

import torch

from milkie.prompt.prompt import Loader
from milkie.utils.data_utils import loadFromYaml

@dataclass
class BaseConfig(ABC):
    pass

from enum import Enum

class DocsetType(Enum):
    LONG_TERM = 0

class LongTermDocsetSource(Enum):
    LOCAL = 0

class LLMType(Enum):
    HUGGINGFACE = 0
    GEN_OPENAI = 1
    AZURE_OPENAI = 2

class FRAMEWORK(Enum):
    NONE = 0

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

class StorageType(Enum):
    FILE = 0
    REDIS = 1

class DocsetTermConfig(BaseConfig):
    def __init__(
            self, 
            type :DocsetType, 
            source :LongTermDocsetSource,
            path :str):
        self.type = type 
        self.source = source
        self.path = path
    
    def fromArgs(config :dict):
        if config["type"] == DocsetType.LONG_TERM.name:
            if config["source"] == LongTermDocsetSource.LOCAL.name:
                return DocsetTermConfig(
                    type=DocsetType.LONG_TERM,
                    source=LongTermDocsetSource.LOCAL,
                    path=config["path"])
            else:
                raise Exception(f"Long term memory source not supported[{config['source']}]")
        else:
            raise Exception(f"Docset type not supported[{config['type']}]")

class DocsetConfig(BaseConfig):
    def __init__(self, docsetConfig :List[DocsetTermConfig]):
        self.docsetConfig = docsetConfig

    def fromArgs(config :dict):
        if not config:
            return None
        
        configs = []
        for singleConfig in config:
            docsetTermConfig = DocsetTermConfig.fromArgs(singleConfig)
            configs.append(docsetTermConfig)
        return DocsetConfig(docsetConfig=configs)

class LLMBasicConfig(BaseConfig):
    def __init__(
            self, 
            systemPrompt :str,
            defaultModel :str, 
            codeModel :list[str], 
            skillModel :str,
            ctxLen :int):
        self.systemPrompt = systemPrompt
        self.defaultModel = defaultModel
        self.codeModel = codeModel
        self.skillModel = skillModel if skillModel else defaultModel
        self.ctxLen = ctxLen

    def fromArgs(config :dict):
        return LLMBasicConfig(
            systemPrompt=config["system_prompt"],
            defaultModel=config["default_model"],
            codeModel=config["code_model"],
            skillModel=config.get("skill_model", None),
            ctxLen=config["ctx_len"])

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
            from transformers import BitsAndBytesConfig
            quantizationConfig = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantizationType == QuantType.INT4:
            from transformers import BitsAndBytesConfig
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
class SingleLLMConfig(BaseConfig):
    def __init__(
            self, 
            type :LLMType, 
            name :str,
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
            reasoner :bool = False,
            modelArgs :LLMModelArgs = None,
            generationArgs :LLMGenerationArgs = None):
        self.type = type
        self.name = name if name else model
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
        self.reasoner = reasoner
        self.modelArgs = modelArgs
        self.generationArgs = generationArgs
    
    def fromArgs(basicConfig :LLMBasicConfig, config :dict, cloud_configs :dict = None):
        framework = FRAMEWORK.NONE
        if "framework" in config.keys():
            if config["framework"] == FRAMEWORK.VLLM.name:
                framework = FRAMEWORK.VLLM
            elif config["framework"] == FRAMEWORK.LMDEPLOY.name:
                framework = FRAMEWORK.LMDEPLOY
            elif config["framework"] == FRAMEWORK.HUGGINGFACE.name:
                framework = FRAMEWORK.HUGGINGFACE

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

        promptName = config["system_prompt"] if "system_prompt" in config else basicConfig.systemPrompt
        ctxLen = config["ctx_len"] if "ctx_len" in config else basicConfig.ctxLen
        reasoner = config.get("reasoner", False)
        
        systemPrompt = Loader.load(promptName)
        if config["type"] == LLMType.HUGGINGFACE.name:
            return SingleLLMConfig(
                type=LLMType.HUGGINGFACE, 
                model=config["model"],
                systemPrompt=systemPrompt,
                ctxLen=ctxLen,
                batchSize=config.get("batch_size", 1),
                tensorParallelSize=config.get("tensor_parallel_size", 1),
                framework=framework,
                device=device,
                port=port,
                reasoner=reasoner,
                modelArgs=modelArgs,
                generationArgs=generationArgs)
        elif config["type"] == LLMType.GEN_OPENAI.name:
            # 使用cloud配置
            source = config.get("source", None)
            api_key = config.get("api_key", None)
            endpoint = config.get("endpoint", None)
            
            if not source:
                raise Exception("source is required for GEN_OPENAI")
            
            # 从cloud_configs中查找相应的配置
            cloud_config = None
            if isinstance(cloud_configs, list):
                cloud_config = next((c for c in cloud_configs if c.source == source), None)
            
            if cloud_config:
                # 如果本地配置中没有提供，则使用云配置中的值
                api_key = api_key or cloud_config.api_key
                endpoint = endpoint or cloud_config.endpoint
            
            # 最终检查必要的参数
            if not api_key or not endpoint:
                raise Exception(f"api_key and endpoint are required for GEN_OPENAI with source '{source}'")
            
            return SingleLLMConfig(
                type=LLMType.GEN_OPENAI,
                name=config.get("name", None),
                model=config["model"],
                systemPrompt=systemPrompt,
                ctxLen=ctxLen,
                apiKey=api_key,
                endpoint=endpoint,
                reasoner=reasoner,
                generationArgs=generationArgs)
        elif config["type"] == LLMType.AZURE_OPENAI.name:
            # 使用cloud配置
            source = config.get("source", None)
            api_key = config.get("api_key", None)
            endpoint = config.get("endpoint", None)
            
            # 如果指定了source且cloud_configs不为空，从cloud配置中获取api_key和endpoint
            if source and cloud_configs and source in cloud_configs:
                cloud_config = next((c for c in cloud_configs if c["source"] == source), None)
                if cloud_config:
                    api_key = cloud_config.get("api_key", api_key)
                    endpoint = cloud_config.get("endpoint", endpoint)
                    
            return SingleLLMConfig(
                type=LLMType.AZURE_OPENAI,
                model=config["model"],
                name=config.get("name", None),
                batchSize=config.get("batch_size", 1),
                deploymentName=config.get("deployment_name", None),
                apiKey=api_key,
                endpoint=endpoint,
                apiVersion=config.get("api_version", None),
                reasoner=reasoner,
                modelArgs=modelArgs,
                generationArgs=generationArgs)
        else:
            raise Exception("LLM type not supported")

class LLMConfig(BaseConfig):

    def __init__(self, llmConfigs :List[SingleLLMConfig]):
        self.llmConfigs = llmConfigs
        self.llmMap = {}
        for llmConfig in llmConfigs:
            self.llmMap[llmConfig.model] = llmConfig
    
    def fromArgs(basicConfig :LLMBasicConfig, config :dict, cloud_configs :list = None):
        configs = []
        modelNames = []
        for provider, provider_configs in config.items():
            for singleConfig in provider_configs:
                singleConfig["source"] = provider
                if "type" not in singleConfig:
                    singleConfig["type"] = LLMType.GEN_OPENAI.name
                        
                llmConfig = SingleLLMConfig.fromArgs(basicConfig, singleConfig, cloud_configs)
                if llmConfig.name not in modelNames:
                    modelNames.append(llmConfig.name)
                else:
                    raise Exception(f"LLM name {llmConfig.name} duplicated")
                configs.append(llmConfig)
        return LLMConfig(configs)

    def getConfig(self, model :str) -> SingleLLMConfig:
        return self.llmMap.get(model)

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

class StorageConfig(BaseConfig):
    def __init__(
            self, 
            type: StorageType, 
            mempath: str,
            flushIntervalSec: int):
        self.type = type
        self.mempath = mempath
        self.flushIntervalSec = flushIntervalSec

    def fromArgs(config :dict):
        type = None
        if config["type"] == StorageType.FILE.name:
            type = StorageType.FILE
        elif config["type"] == StorageType.REDIS.name:
            type = StorageType.REDIS
        else:
            raise Exception(f"Storage type not supported[{config['type']}]")

        return StorageConfig(
            type=type, 
            mempath=config["mempath"],
            flushIntervalSec=config["flush_interval_sec"])

class KnowhowConfig(BaseConfig):
    def __init__(self, maxNum: int):
        self.maxNum = maxNum

    def fromArgs(config :dict):
        return KnowhowConfig(maxNum=config["max_num"])

class ExperienceConfig(BaseConfig):
    def __init__(self, maxNum: int):
        self.maxNum = maxNum

    def fromArgs(config :dict):
        return ExperienceConfig(maxNum=config["max_num"])

class MemoryConfig(BaseConfig):
    def __init__(
            self, 
            storageConfig: StorageConfig,
            knowhowConfig: KnowhowConfig,
            experienceConfig: ExperienceConfig):
        self.storageConfig = storageConfig
        self.knowhowConfig = knowhowConfig
        self.experienceConfig = experienceConfig

    def fromArgs(config :dict):
        return MemoryConfig(
            storageConfig=StorageConfig.fromArgs(config["storage"]),
            knowhowConfig=KnowhowConfig.fromArgs(config["knowhow"]),
            experienceConfig=ExperienceConfig.fromArgs(config["experience"]))

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

class CloudSourceConfig(BaseConfig):
    def __init__(self, source: str, api_key: str, endpoint: str, additional_params: dict = None):
        self.source = source
        self.api_key = api_key
        self.endpoint = endpoint
        self.additional_params = additional_params or {}
    
    @classmethod
    def fromArgs(cls, config: dict):
        if not config.get("source"):
            raise Exception("Cloud source config must have a 'source' field")
        return cls(
            source=config["source"],
            api_key=config.get("api_key"),
            endpoint=config.get("endpoint"),
            additional_params={k: v for k, v in config.items() 
                              if k not in ["source", "api_key", "endpoint"]}
        )

class CloudConfig(BaseConfig):
    def __init__(self, cloud_sources: List[CloudSourceConfig]):
        self.cloud_sources = cloud_sources
        self.source_map = {source.source: source for source in cloud_sources}
    
    @classmethod
    def fromArgs(cls, config: list):
        if not config:
            return cls([])
        
        sources = []
        for source_config in config:
            sources.append(CloudSourceConfig.fromArgs(source_config))
        return cls(sources)
    
    def getSourceConfig(self, source: str) -> CloudSourceConfig:
        return self.source_map.get(source)

class VMConnectionType(Enum):
    SSH = 0
    DOCKER = 1

class VMConfig(BaseConfig):
    def __init__(
            self, 
            connectionType: VMConnectionType, 
            host: str, 
            port: int, 
            username: str, 
            encryptedPassword: str,
            sshKeyPath: str = None,
            timeout: int = 10,
            retryCount: int = 3):
        self.connectionType = connectionType
        self.host = host
        self.port = port
        self.username = username
        self.encryptedPassword = encryptedPassword
        self.sshKeyPath = sshKeyPath
        self.timeout = timeout
        self.retryCount = retryCount

    @staticmethod
    def fromArgs(config: dict):
        # 获取基本配置
        connectionType = VMConnectionType.SSH if config["connection_type"] == "ssh" else VMConnectionType.DOCKER
        host = config["host"]
        port = config["port"]
        username = config["username"]
        encryptedPassword = config["encrypted_password"]
        
        # 获取可选配置
        sshKeyPath = config.get("ssh_key_path", None)
        timeout = config.get("timeout", 10)
        retryCount = config.get("retry_count", 3)
        
        return VMConfig(
            connectionType=connectionType,
            host=host,
            port=port,
            username=username,
            encryptedPassword=encryptedPassword,
            sshKeyPath=sshKeyPath,
            timeout=timeout,
            retryCount=retryCount
        )
        
    def validate(self) -> bool:
        """验证配置是否有效
        
        Returns:
            bool: 配置是否有效
        """
        # 验证基本参数
        if not self.host or not isinstance(self.host, str):
            return False
        
        if not isinstance(self.port, int) or self.port <= 0 or self.port > 65535:
            return False
            
        if not self.username or not isinstance(self.username, str):
            return False
            
        # 密码和SSH密钥至少要有一个
        if not self.encryptedPassword and not self.sshKeyPath:
            return False
            
        # 如果指定了SSH密钥路径，检查文件是否存在
        if self.sshKeyPath and not os.path.exists(self.sshKeyPath):
            return False
            
        return True
        
    def toDict(self) -> dict:
        """将配置转换为字典
        
        Returns:
            dict: 配置字典
        """
        return {
            "connection_type": "ssh" if self.connectionType == VMConnectionType.SSH else "docker",
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "encrypted_password": self.encryptedPassword,
            "ssh_key_path": self.sshKeyPath,
            "timeout": self.timeout,
            "retry_count": self.retryCount
        }
        
    def __str__(self) -> str:
        """返回配置的字符串表示
        
        Returns:
            str: 配置的字符串表示
        """
        return f"VMConfig(type={self.connectionType.name}, host={self.host}, port={self.port}, username={self.username})"

class DataSourceType(Enum):
    MYSQL = 0
    # 可以在未来添加更多数据源类型

class DataSourceConfig(BaseConfig):
    def __init__(
            self, 
            name: str,
            type: DataSourceType,
            host: str,
            port: int,
            username: str,
            password: str,
            database: str,
            additional_params: dict = None):
        self.name = name
        self.type = type
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.additional_params = additional_params or {}

    @classmethod
    def fromArgs(cls, config: dict):
        if not config:
            return None
            
        name = config.get("name")
        typeStr = config.get("type")
        
        if not name or not typeStr:
            raise Exception("数据源配置缺少必要字段: name 或 type")
            
        # 将类型字符串转换为 DataSourceType 枚举
        try:
            dataSourceType = DataSourceType[typeStr.upper()]
        except KeyError:
            raise Exception(f"不支持的数据源类型: {typeStr}")
            
        # 获取基本配置
        host = config.get("host")
        port = config.get("port")
        username = config.get("username")
        password = config.get("password")
        database = config.get("database")
        
        # 获取其他可能的参数
        additional_params = {}
        for key, value in config.items():
            if key not in ["name", "type", "host", "port", "username", "password", "database"]:
                additional_params[key] = value
                
        return cls(
            name=name,
            type=dataSourceType,
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            additional_params=additional_params
        )

class DataSourcesConfig(BaseConfig):
    def __init__(self, dataSourceConfigs: List[DataSourceConfig]):
        self.dataSourceConfigs = dataSourceConfigs
        self.source_map = {ds.name: ds for ds in dataSourceConfigs} if dataSourceConfigs else {}
    
    @classmethod
    def fromArgs(cls, config: list):
        if not config:
            return cls([])
            
        sources = []
        for source_config in config:
            sources.append(DataSourceConfig.fromArgs(source_config))
            
        return cls(sources)
    
    def getSourceConfig(self, name: str) -> DataSourceConfig:
        return self.source_map.get(name)

    def getAllSourceConfigs(self) -> List[DataSourceConfig]:
        return self.dataSourceConfigs

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
        # 统一使用fromArgs模式，必选项使用索引操作，可选项使用条件表达式
        self.cloudConfig = CloudConfig.fromArgs(config.get("cloud", []))
        self.llmBasicConfig = LLMBasicConfig.fromArgs(config["llm_basic"])
        self.llmConfig = LLMConfig.fromArgs(
            self.llmBasicConfig, config["llm"], self.cloudConfig.cloud_sources
        )
        
        # 统一可选配置的处理方式
        self.embeddingConfig = (
            EmbeddingConfig.fromArgs(config["embedding"]) 
            if "embedding" in config 
            else None
        )
        
        self.docsetConfig = DocsetConfig.fromArgs(config.get("docset", []))
        self.indexConfig = IndexConfig.fromArgs(config.get("index", {}))
        self.retrievalConfig = (
            RetrievalConfig.fromArgs(config["retrieval"]) 
            if "retrieval" in config 
            else None
        )
        self.memoryConfig = MemoryConfig.fromArgs(config.get("memory", {}))
        self.toolsConfig = ToolsConfig.fromArgs(config.get("tools", {}))
        self.vmConfig = VMConfig.fromArgs(config.get("vm", {}))
        self.dataSourcesConfig = DataSourcesConfig.fromArgs(config.get("datasources", []))

    def getCloudConfig(self, source: str) -> CloudSourceConfig:
        """获取特定云服务提供商的配置"""
        return self.cloudConfig.getSourceConfig(source)

    def getCloudConfigs(self) -> List[CloudSourceConfig]:
        """获取所有云服务提供商配置"""
        return self.cloudConfig.cloud_sources
                    
    def getLLMBasicConfig(self) -> LLMBasicConfig:
        return self.llmBasicConfig

    def getLLMConfig(self) -> LLMConfig:
        return self.llmConfig

    def getSingleLLMConfig(self, model :str) -> SingleLLMConfig:
        return self.llmConfig.getConfig(model)

    def getDefaultLLMConfig(self) -> SingleLLMConfig:
        return self.getSingleLLMConfig(self.llmBasicConfig.defaultModel)

    def getEmbeddingConfig(self) -> EmbeddingConfig:
        return self.embeddingConfig

    def getDocsetConfig(self) -> DocsetConfig:
        return self.docsetConfig

    def getIndexConfig(self) -> IndexConfig:
        return self.indexConfig

    def getRetrievalConfig(self) -> RetrievalConfig:
        return self.retrievalConfig
    
    def getMemoryConfig(self) -> MemoryConfig:
        return self.memoryConfig

    def getToolsConfig(self) -> ToolsConfig:
        return self.toolsConfig
    
    def getEmailConfig(self) -> EmailConfig:
        return self.toolsConfig.getEmailConfig()
    
    def getVMConfig(self) -> VMConfig:
        return self.vmConfig

    def getDataSourcesConfig(self) -> DataSourcesConfig:
        """获取所有数据源配置"""
        return self.dataSourcesConfig

    def getDataSourceConfig(self, name: str) -> DataSourceConfig:
        """获取特定名称的数据源配置"""
        return self.dataSourcesConfig.getSourceConfig(name)