from typing import Sequence
import torch
import logging, json
from dataclasses import asdict
from llama_index import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.types import ChatMessage

from milkie.prompt.prompt import Loader
from milkie.config.config import EmbeddingConfig, LLMConfig

from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)

SystemPromptCn = Loader.load("system_prompt")

def tokenizerMessagesToPrompt(self, messages: Sequence[ChatMessage]) -> str:
    if hasattr(self._tokenizer, "apply_chat_template"):
        messages_dict = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        tokens = self._tokenizer.apply_chat_template(
            messages_dict, 
            generic_messages_to_prompt=True)
        return self._tokenizer.decode(tokens)

    return generic_messages_to_prompt(messages)

class ModelFactory:
    
    def __init__(self) -> None:
        self.models = {}

    def getLLM(self, config :LLMConfig):
        if config.model not in self.models:
            self.models[config.model] = HuggingFaceLLM(
                context_window=config.ctxLen,
                max_new_tokens=256,
                model_kwargs={"torch_dtype":torch.bfloat16, "trust_remote_code": True},
                generate_kwargs={"temperature": config.temperature, "do_sample": False},
                system_prompt=SystemPromptCn,
                query_wrapper_prompt=PromptTemplate("{query_str}\n<|ASSISTANT|>\n"),
                tokenizer_name=config.model,
                model_name=config.model,
                messages_to_prompt=tokenizerMessagesToPrompt,
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": config.ctxLen, "use_fast": False, "trust_remote_code": True},
                is_chat_model=True,
            )
        logging.info(f"Building HuggingFaceLLM with model {config.model} from_cache{repr in self.models}")
        self.models[config.model].generate_kwargs["temperature"] = config.temperature
        return self.models[config.model]

    def getEmbedding(self, config :EmbeddingConfig):
        if config.model not in self.models:
            self.models[config.model] = HuggingFaceEmbedding(
                model_name=config.model,
                device=config.device)
        logging.info(f"Building HuggingFaceEmbedding with model {config.model} from_cache{repr in self.models}")
        return self.models[config.model]