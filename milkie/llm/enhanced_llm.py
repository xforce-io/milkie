from abc import abstractmethod
from typing import Any, Sequence
from llama_index_client import ChatMessage
from llama_index.legacy.core.llms.types import ChatResponse
import torch
from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
)

class EnhancedLLM(object):

    def __init__(self) -> None:
        self._model = None

    def getMem(self) -> float:
        return round(self._model.get_memory_footprint()/(1024*1024*1024), 2)

    def getNumParams(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    #get memory bandwidth utilization
    def getMBU(self, tokensPerSec :float, memBandwidth :float) -> float:
        return self.getNumParams() * self.__getSingleParameterSizeInBytes() * tokensPerSec / memBandwidth

    def __chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._model.messages_to_prompt(messages)
        completion_response = self.__complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @abstractmethod
    def __complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        pass

    def __getSingleParameterSizeInBytes(self):
        type_to_size = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
        }

        dtype = self._model.dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size

