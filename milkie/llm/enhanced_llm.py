from abc import abstractmethod
from typing import Any, Sequence
from llama_index_client import ChatMessage
from llama_index.legacy.core.llms.types import ChatResponse
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.legacy.llms.llm import LLM
import torch
from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
)

class EnhancedLLM(object):

    def __init__(self) -> None:
        self._llm :LLM = None

    def getLLM(self) -> LLM:
        return self._llm

    def getMem(self) -> float:
        return -1

    def getNumParams(self) -> int:
        return 0

    #get memory bandwidth utilization
    def getMBU(self, tokensPerSec :float, memBandwidth :float) -> float:
        return self.getNumParams() * self._getSingleParameterSizeInBytes() * tokensPerSec / memBandwidth

    @torch.inference_mode()
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any):
        messages = self._llm._get_messages(prompt, **prompt_args)
        response = self._chat(messages)
        output = response.message.content or ""
        return (self._llm._parse_output(output), len(response.raw["model_output"]))

    @torch.inference_mode()
    def predictBatch(
            self, 
            prompt: BasePromptTemplate, 
            argsList: list[dict]):
        result = []
        responses = self._chatBatch([self._llm._get_messages(prompt, **args) for args in argsList])
        for response in responses:
            output = response.message.content or ""
            result += [(self._llm._parse_output(output), len(response.raw["model_output"]))]
        return result

    @abstractmethod
    def _getModel(self):
        pass

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._llm.messages_to_prompt(messages)
        completion_response = self._complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    def _chatBatch(self, messagesBatch: list[Sequence[ChatMessage]], **kwargs: Any) -> list[ChatResponse]:
        prompts = []
        for messages in messagesBatch:
            prompts += [self._llm.messages_to_prompt(messages)]
        completionResponses = self._completeBatch(prompts, formatted=True, **kwargs)
        return [completion_response_to_chat_response(completionResponse) for completionResponse in completionResponses]

    @abstractmethod
    def _complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _completeBatch(self, prompts: list[str], formatted: bool = False, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _getSingleParameterSizeInBytes(self):
        pass