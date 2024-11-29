import logging

from llama_index.legacy.llms import AzureOpenAI

from milkie.config.config import EmbeddingConfig, GlobalConfig, LLMConfig, SingleLLMConfig, LLMType
from milkie.model_factory import ModelFactory
from milkie.prompt.prompt import Loader

class Settings(object):
    def __init__(
            self, 
            config :GlobalConfig,
            modelFactory :ModelFactory) -> None:
        self.modelFactory = modelFactory
        self.llmBasicConfig = config.getLLMBasicConfig()
        self.llms = self._buildLLMs(config.getLLMConfig())
        self.llmDefault = self.getLLM(self.llmBasicConfig.defaultModel)
        self.llmCode = self.getLLM(self.llmBasicConfig.codeModel)
        if config.embeddingConfig:
            self._buildEmbedding(config.embeddingConfig)
        else:
            self.embedding = None

    def getAllLLMs(self):
        return self.llms.keys()

    def getLLM(self, name :str):
        return self.llms[name]
    
    def getLLMCode(self):
        return self.llmCode
    
    def getLLMDefault(self):
        return self.llmDefault

    def _buildLLMs(self, config :LLMConfig):
        llms = {}
        for singleConfig in config.llmConfigs:
            llms[singleConfig.name] = self._buildSingleLLM(singleConfig)
        return llms

    def _buildSingleLLM(self, config :SingleLLMConfig):
        if config.type == LLMType.HUGGINGFACE or config.type == LLMType.GEN_OPENAI:
            return self.modelFactory.getLLM(config)
        elif config.type == LLMType.AZURE_OPENAI:
            logging.info(f"Building AzureOpenAI with model {config.model}")
            return AzureOpenAI(
                azure_endpoint=config.endpoint,
                azure_deployment=config.deploymentName,
                api_version=config.apiVersion,
                api_key=config.apiKey,
                system_prompt=Loader.load("system_prompt"),
                temperature=config.temperature)
        return None

    def _buildEmbedding(self, config :EmbeddingConfig):
        self.embedding = self.modelFactory.getEmbedding(config) 
