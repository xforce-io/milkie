import torch
import logging, json
from dataclasses import asdict
from llama_index import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding

from milkie.prompt.prompt import Loader
from milkie.config.config import EmbeddingConfig, LLMConfig

SystemPromptCn = Loader.load("system_prompt")

class ModelFactory:
    
    def __init__(self) -> None:
        self.models = {}

    def getLLM(self, config :LLMConfig):
        repr = self.__getReprFromConfig(config)
        if repr not in self.models:
            self.models[config.model] = HuggingFaceLLM(
                context_window=config.ctxLen,
                max_new_tokens=256,
                model_kwargs={"torch_dtype":torch.bfloat16},
                generate_kwargs={"temperature": config.temperature, "do_sample": False},
                system_prompt=SystemPromptCn,
                query_wrapper_prompt=PromptTemplate("{query_str}\n<|ASSISTANT|>\n"),
                tokenizer_name=config.model,
                model_name=config.model,
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": config.ctxLen, "use_fast": False},
            )
        logging.info(f"Building HuggingFaceLLM with model {config.model} from_cache{repr in self.models}")
        return self.models[repr]

    def getEmbedding(self, config :EmbeddingConfig):
        repr = self.__getReprFromConfig(config)
        if repr not in self.models:
            self.embedding = HuggingFaceEmbedding(
                model_name=config.model,
                device=config.device)
        logging.info(f"Building HuggingFaceEmbedding with model {config.model} from_cache{repr in self.models}")

    def __getReprFromConfig(self, config):
        return json.dumps(asdict(config))