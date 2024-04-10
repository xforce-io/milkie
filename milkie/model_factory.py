from typing import Sequence
import logging
from llama_index.legacy.prompts.base import PromptTemplate
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.core.llms.types import MessageRole, ChatMessage
from llama_index.legacy.utils import truncate_text

from milkie.llm.enhanced_hf_llm import EnhancedHFLLM
from milkie.llm.enhanced_vllm import EnhancedVLLM
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
        self.llm = None
        self.signatureLLMModel = None
        self.embedModel = {}

    def getLLM(self, config :LLMConfig):
        signatureModel = self.__getSignatureModel(config)
        if signatureModel == self.signatureLLMModel:
            return self.llm

        llm = self.__setLLMModel(config)
        self.signatureLLMModel = signatureModel
        return llm

    def getEmbedding(self, config :EmbeddingConfig):
        logging.info(f"Building HuggingFaceEmbedding with model {config.model} from_cache[{config.model in self.embedModel}]")
        if config.model not in self.embedModel:
            self.embedModel[config.model] = HuggingFaceEmbedding(
                model_name=config.model,
                device=config.device)
        return self.embedModel[config.model]

    def __setLLMModel(self, config :LLMConfig):
        if self.llm is not None:
            del self.llm
            self.llm = None

        self.llm = EnhancedHFLLM(
            context_window=config.ctxLen,
            max_new_tokens=256,
            model_kwargs=config.modelArgs.toJson(),
            generate_kwargs=config.generationArgs.toJson(),
            system_prompt=SystemPromptCn,
            query_wrapper_prompt=PromptTemplate("{query_str}\n<|ASSISTANT|>\n"),
            tokenizer_name=config.model,
            model_name=config.model,
            messages_to_prompt=messagesToPrompt,
            tokenizer_kwargs={"max_length": config.ctxLen, "use_fast": False, "trust_remote_code": True},
            is_chat_model=True,
        )
        self.llm = EnhancedVLLM(
            model_name=config.model,
            max_new_tokens=256,
            message_to_prompt=messagesToPrompt)
        logging.info(f"Building HuggingFaceLLM with model {config.model} model_args[{repr(config.modelArgs)}] memory[{self.llm.getMem()}GB]")
        return self.llm
    
    def __getSignatureModel(self, config :LLMConfig):
        return "%s-%s" % (config.model, config.modelArgs.toJson())