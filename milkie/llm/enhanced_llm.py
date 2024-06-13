from abc import abstractmethod
from typing import Any, Optional, Sequence
from threading import Thread
import random, uuid
from queue import Queue
from typing import Callable

import time
from pydantic import Field, PrivateAttr
import torch
import logging

from transformers import AutoTokenizer

from llama_index_client import ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core import BasePromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.base.llms.types import CompletionResponseGen

from milkie.config.config import QuantMethod
from milkie.prompt.prompt import Loader

logger = logging.getLogger(__name__)

class LLMApi(CustomLLM):

    model: Optional[str] = Field(description="The Model to use.")

    context_window: Optional[int] = Field(
        default=8192,
    )

    client: Any = PrivateAttr()

    def __init__(self, 
            context_window :int,
            model_name :str,
            client :Any):
        super().__init__(model=model_name)

        self.context_window = context_window
        self.model = model_name
        self.client = client

    def getClient(self):
        return self.client

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self.context_window,)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise (ValueError("Not Implemented")) 

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))

class QueueRequest:
    def __init__(
            self, 
            requestId :str,
            prompt: str, 
            tokenized: list[int],
            **kwargs: Any) -> None:
        self.requestId = str(uuid.uuid4()) if requestId is None else requestId
        self.sessionid = random.randint(0, 2**16)
        self.prompt = prompt
        self.tokenized = tokenized
        self.kwargs = kwargs
    
class QueueResponse:
    def __init__(self,
            requestId :int,
            output :Any) -> None:
        self.requestId = requestId
        self.output = output
        
class EnhancedLLM(object):

    def __init__(self,
            context_window :int,
            concurrency :int,
            tensor_parallel_size :int,
            tokenizer_name :str,
            model_name :str,
            system_prompt :str,
            device :str,
            port :int,
            tokenizer_kwargs :dict) -> None:
        self.context_window = context_window
        self.concurrency = concurrency
        self.tensor_parallel_size = tensor_parallel_size
        self.model_name = model_name
        self.device = device
        self.port = port

        self._llm :LLM = None

        if tokenizer_name:
            self._initTokenizer(tokenizer_name, tokenizer_kwargs)
        else:
            self._tokenizer = None

        self._systemPrompt = Loader.load(system_prompt) if system_prompt is not None else None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=self.context_window,)

    def getLLM(self) -> LLM:
        return self._llm

    def getMem(self) -> float:
        return -1

    def getNumParams(self) -> int:
        return 0

    def getQuantMethod(modelName :str) -> QuantMethod:
        if modelName.lower().find("gptq") >= 0:
            return QuantMethod.GPTQ
        elif modelName.lower().find("awq") >= 0:
            return QuantMethod.AWQ
        else:
            return QuantMethod.NONE

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
            argsList: list[dict],
            **kwargs: Any):
        if argsList is not None and len(argsList) > 0:
            messages = [self._llm._get_messages(prompt, **args) for args in argsList]
        else:
            messages = [self._llm._get_messages(prompt, **{})]
        return self._predictBatch(messages, **kwargs)

    def filterGenArgs(kwargs :dict):
        return EnhancedLLM.filterArgs(kwargs, ["repetition_penalty", "temperature", "top_k", "top_p"])

    def filterArgs(kwargs :dict, keysLeft :list[str]):
        return {k: v for k, v in kwargs.items() if k in keysLeft} 

    @abstractmethod
    def _getModel(self):
        pass

    def _initTokenizer(self, tokenizer_name :str, tokenizer_kwargs :dict):
        tokenizer_kwargs["padding_side"] = "left"
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    
    def _unpadTokenized(tokenized :dict, idx :int) -> list[int]:
        lenMasked = tokenized["attention_mask"][idx].sum(dim=0)
        return tokenized["input_ids"][idx][-lenMasked:]

    def _tokenizer_messages_to_prompt(self, messagesBatch: list[Sequence[ChatMessage]]) -> list[str]:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            messagesDict = []
            for messages in messagesBatch:
                messagesDict += [[
                    {"role": message.role.value, "content": message.content}
                    for message in messages
                ]]
            return self._tokenizer.apply_chat_template(
                messagesDict,
                add_generation_prompt=True,
                tokenize=False)
        return [generic_messages_to_prompt(messages) for messages in messagesBatch]

    @torch.inference_mode()
    def _predictBatch(
            self, 
            messages,
            **kwargs: Any):
        result = []
        if self._systemPrompt is not None:
            for msgs in messages:
                msgs.insert(0, ChatMessage(role="system", content=self._systemPrompt))
        responses = self._chatBatch(messages, **kwargs)
        for response in responses:
            output = response.message.content or ""
            result += [(self._llm._parse_output(output), len(response.raw["model_output"]))]
        return result

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._llm.messages_to_prompt(messages)
        completion_response = self._complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    def _chatBatch(
            self, 
            messagesBatch: list[Sequence[ChatMessage]], 
            **kwargs: Any) -> list[ChatResponse]:
        prompts = self._tokenizer_messages_to_prompt(messagesBatch)

        t0 = time.time()
        completionResponses = self._completeBatch(prompts, **kwargs)
        t1 = time.time()

        firstPrompt = prompts[0].replace("\n", "//")
        logger.debug(f"size[{len(prompts)}] " 
            f"prompt[{firstPrompt}] "
            f"answer[{completionResponses[0].text}]"
            f"costMs[{t1-t0}]")

        return [completion_response_to_chat_response(completionResponse) for completionResponse in completionResponses]

    @abstractmethod
    def _complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _completeBatch(self, prompts: list[str], **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _getSingleParameterSizeInBytes(self):
        pass

    def _completeBatchAsync(
            self, 
            prompts: list[str], 
            numThreads: int,
            inference: Callable[[object, Queue[QueueRequest], Queue[QueueResponse], dict, dict], Any],
            tokenIdExtractor: Callable[[QueueResponse], list[int]],
            **genArgs: Any
    ) -> CompletionResponse:

        reqQueue = Queue[QueueRequest]()
        resQueue = Queue[QueueResponse]()
        threads = []
        
        inputs = self._tokenizer(text=prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        order = dict()
        for i, prompt in enumerate(prompts):
            unpaddedInputIds = EnhancedLLM._unpadTokenized(inputs, i)
            request = QueueRequest(
                    requestId=None,
                    prompt=prompt, 
                    tokenized=unpaddedInputIds.tolist(),
                    **genArgs)
            order[request.requestId] = i
            reqQueue.put(request)

        for i in range(numThreads):
            reqQueue.put(None)
        
        for i in range(numThreads):
            t = Thread(
                    target=inference, 
                    args=(self, reqQueue, resQueue, genArgs), 
                    daemon=True)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        resps :list[QueueResponse] = []
        while not resQueue.empty():
            resps.append(resQueue.get())
        resps.sort(key=lambda x: order[x.requestId])

        assert len(resps) == len(prompts)

        completionTokens = []
        for resp in resps:
            completionTokens += [tokenIdExtractor(resp.output)]
        completion = self._tokenizer.batch_decode(completionTokens, skip_special_tokens=True)

        completionResponses = []
        for i, resp in enumerate(resps):
            completionResponses += [CompletionResponse(
                text=completion[i], 
                raw={"model_output": tokenIdExtractor(resp.output)})]
        return completionResponses

    def _completeBatchNoTokenizationAsync(
            self, 
            prompts: list[str], 
            numThreads: int,
            inference: Callable[[object, Queue[QueueRequest], Queue[QueueResponse], dict, dict], Any],
            **genArgs: Any
    ) -> CompletionResponse:

        reqQueue = Queue[QueueRequest]()
        resQueue = Queue[QueueResponse]()
        threads = []
        
        order = dict()
        for i, prompt in enumerate(prompts):
            request = QueueRequest(
                    requestId=None,
                    prompt=prompt, 
                    tokenized=None,
                    **genArgs)
            order[request.requestId] = i
            reqQueue.put(request)

        for i in range(numThreads):
            reqQueue.put(None)
        
        for i in range(numThreads):
            t = Thread(
                    target=inference, 
                    args=(self, reqQueue, resQueue, genArgs), 
                    daemon=True)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        resps :list[QueueResponse] = []
        while not resQueue.empty():
            resps.append(resQueue.get())
        resps.sort(key=lambda x: order[x.requestId])

        assert len(resps) == len(prompts)

        completionResponses = []
        for i, resp in enumerate(resps):
            completionResponses += [CompletionResponse(text=resp.output)]
        return completionResponses