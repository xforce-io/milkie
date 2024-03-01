from typing import Sequence
import torch
import logging, json
from llama_index import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.types import MessageRole, ChatMessage
from llama_index.indices.utils import truncate_text

from milkie.prompt.prompt import Loader
from milkie.config.config import EmbeddingConfig, LLMConfig

logger = logging.getLogger(__name__)

SystemPromptCn = Loader.load("system_prompt")

def messagesToPrompt(messages: Sequence[ChatMessage]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
    result = "\n".join(string_messages)

    fmt_text_chunk = truncate_text(result, 5000).replace("\n", "//")
    logger.debug(f"> prompt: [{len(fmt_text_chunk)}|{fmt_text_chunk}]")

    return result

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
                messages_to_prompt=messagesToPrompt,
                device_map="auto",
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