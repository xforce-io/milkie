import logging
from llama_index import PromptTemplate
import torch
from milkie.config.config import EmbeddingConfig, GlobalConfig, LLMConfig, LLMType

from llama_index.llms import HuggingFaceLLM, AzureOpenAI

from milkie.model_factory import ModelFactory
from milkie.prompt.prompt import Loader

SystemPromptCn = Loader.load("system_prompt")

class Settings(object):
    def __init__(
            self, 
            config :GlobalConfig,
            modelFactory :ModelFactory) -> None:
        self.modelFactory = modelFactory
        self.__buildLLM(config.llmConfig)
        self.__buildEmbedding(config.embeddingConfig)

    def __buildLLM(self, config :LLMConfig):
        if config.type == LLMType.HUGGINGFACE:
            self.llm = self.modelFactory.getLLM(config)
        elif config.type == LLMType.AZURE_OPENAI:
            logging.info(f"Building AzureOpenAI with model {config.model}")
            self.llm = AzureOpenAI(
                azure_endpoint=config.azureEndpoint,
                azure_deployment=config.deploymentName,
                api_version=config.apiVersion,
                api_key=config.apiKey,
                system_prompt=SystemPromptCn,
                temperature=config.temperature)

    def __buildEmbedding(self, config :EmbeddingConfig):
        self.embedding = self.modelFactory.getEmbedding(config) 
