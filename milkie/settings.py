import logging
from llama_index import PromptTemplate
import torch
from milkie.config.config import EmbeddingConfig, GlobalConfig, LLMConfig, LLMType

from llama_index.llms import HuggingFaceLLM, AzureOpenAI
from llama_index.embeddings import HuggingFaceEmbedding

from milkie.prompt.prompt import Loader

SystemPromptCn = Loader.load("system_prompt")
logger = logging.getLogger(__name__)

class Settings(object):
    def __init__(self, config :GlobalConfig) -> None:
        self.__buildLLM(config.llmConfig)
        self.__buildEmbedding(config.embeddingConfig)

    def __buildLLM(self, config :LLMConfig):
        if config.type == LLMType.HUGGINGFACE:
            logging.info(f"Building HuggingFaceLLM with model {config.model}")
            self.llm = HuggingFaceLLM(
                context_window=config.ctxLen,
                max_new_tokens=256,
                model_kwargs={"torch_dtype":torch.bfloat16, "trust_remote_code" :True},
                generate_kwargs={"temperature": config.temperature, "do_sample": False},
                system_prompt=SystemPromptCn,
                query_wrapper_prompt=PromptTemplate("{query_str}\n<|ASSISTANT|>\n"),
                tokenizer_name=config.model,
                model_name=config.model,
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": config.ctxLen, "use_fast": False, "trust_remote_code": True},
            )
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
        logging.info(f"Building HuggingFaceEmbedding with model {config.model}")
        self.embedding = HuggingFaceEmbedding(
            model_name=config.model,
            device=config.device)
