from abc import abstractmethod
from typing import Any, Sequence
from llama_index_client import ChatMessage
from llama_index.legacy.core.llms.types import ChatResponse
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.llms.llm import LLM
import torch
from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
)

class EnhancedLLM(object):

    def __init__(self) -> None:
        self._llm :LLM = None

    def getLLM(self) -> LLM:
        return self._llm

    @abstractmethod
    def getMem(self) -> float:
        return -1

    def getNumParams(self) -> int:
        return sum(p.numel() for p in self._getModel().parameters())

    #get memory bandwidth utilization
    def getMBU(self, tokensPerSec :float, memBandwidth :float) -> float:
        return self.getNumParams() * self._getSingleParameterSizeInBytes() * tokensPerSec / memBandwidth

    @torch.inference_mode()
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any):
        if self._llm.metadata.is_chat_model:
            messages = self._llm._get_messages(prompt, **prompt_args)
            response = self._chat(messages)
            output = response.message.content or ""
        else:
            raise NotImplementedError("predict not implemented for non-chat models")
        
        return (self._llm._parse_output(output), len(response.raw["model_output"]))

    @abstractmethod
    def _getModel(self):
        return None

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._llm.messages_to_prompt(messages)
        completion_response = self._complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @abstractmethod
    def _complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        pass

    def _getSingleParameterSizeInBytes(self):
        type_to_size = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
        }

        dtype = self._getModel().dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size

