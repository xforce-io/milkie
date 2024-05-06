import uuid
import random
from queue import Queue
from threading import Thread
from typing import Any, Optional, Sequence
from milkie.llm.enhanced_llm import EnhancedLLM
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import (TurbomindEngineConfig, GenerationConfig, EngineGenerationConfig, EngineOutput)

from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.legacy.bridge.pydantic import Field, PrivateAttr

class LMDeploy(CustomLLM):

    model: Optional[str] = Field(description="The HuggingFace Model to use.")
    _turboMind: Any = PrivateAttr()
    
    def __init__(
            self, 
            model_name: str,
            context_window :int) -> None:
        engineConfig = TurbomindEngineConfig(
            cache_max_entry_count=0.8,
            cache_block_seq_len=64,
            model_format="hf",
            session_len=context_window,
            tp=1)
        super().__init__(model=model_name)
        self._turboMind = TurboMind.from_pretrained(model_name, engineConfig)
        
    def modelInst(self):
        return self._turboMind.create_instance()

    def class_name(cls) -> str:
        return "LMDeploy"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        kwargs = kwargs if kwargs else {}
        return self.chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise (ValueError("Not Implemented"))

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise (ValueError("Not Implemented"))

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise (ValueError("Not Implemented"))

class QueueRequest:
    def __init__(
            self, 
            prompt: str, 
            tokenized: list[int],
            **kwargs: Any) -> None:
        self._uuid = str(uuid.uuid4())
        self.sessionid = random.randint(0, 2**16)
        self.prompt = prompt
        self.tokenized = tokenized
        self.kwargs = kwargs
    
    def uuid(self):
        return self._uuid

class QueueResponse:
    def __init__(self,
            request :QueueRequest,
            output :EngineOutput) -> None:
        self.request = request
        self.output = output
        
    def uuid(self):
        return self.request.uuid()

class EnhancedLmDeploy(EnhancedLLM):
    def __init__(
            self, 
            context_window: int, 
            concurrency: int,
            tokenizer_name: str, 
            model_name: str,
            device: str,
            max_new_tokens: int,
            tokenizer_kwargs: dict) -> None:
        super().__init__(context_window, concurrency, tokenizer_name, device, tokenizer_kwargs)

        self._llm = LMDeploy(
                model_name, 
                context_window)
        self.device = device
        self.maxNewTokens = max_new_tokens
        self.reqQueue = Queue[QueueRequest]()
        self.resQueue = Queue[QueueResponse]()
        self.threads = []

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> CompletionResponse:
        self.reqQueue.queue.clear()
        self.resQueue.queue.clear()
        self.threads.clear()
        
        inputs = self._tokenizer(text=prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        order = dict()
        for i, prompt in enumerate(prompts):
            unpaddedInputIds = inputs["input_ids"][i][:inputs["attention_mask"][i].sum(dim=0)]
            request = QueueRequest(
                    prompt, 
                    unpaddedInputIds.tolist(),
                    **kwargs)
            order[request.uuid()] = i
            self.reqQueue.put(request)

        for i in range(self.concurrency):
            self.reqQueue.put(None)
        
        for i in range(self.concurrency):
            t = Thread(
                    target=EnhancedLmDeploy._inferenceThread, 
                    args=(self, self.reqQueue, self.resQueue), 
                    daemon=True)
            t.start()
            self.threads.append(t)
        
        for t in self.threads:
            t.join()

        resps :list[QueueResponse] = []
        while not self.resQueue.empty():
            resps.append(self.resQueue.get())
        resps.sort(key=lambda x: order[x.uuid()])

        assert len(resps) == len(prompts)

        completionTokens = []
        for resp in resps:
            completionTokens += [resp.output.token_ids]
        completion = self._tokenizer.batch_decode(completionTokens, skip_special_tokens=True)

        completionResponses = []
        for i, resp in enumerate(resps):
            completionResponses += [CompletionResponse(
                text=completion[i], 
                raw={"model_output": resp.output.token_ids})]
        return completionResponses

    def _inferenceThread(
            self, 
            reqQueue :Queue[QueueRequest], 
            resQueue :Queue[QueueResponse], 
            **kwargs :Any) -> EngineOutput:
        genConfig = EngineGenerationConfig.From(
            GenerationConfig(
                n=1,
                max_new_tokens=self.maxNewTokens),
            self._tokenizer
        )

        modelInst = self._llm.modelInst()
        for request in iter(reqQueue.get, None):
            for outputs in modelInst.stream_infer(
                    request.sessionid,
                    input_ids=request.tokenized,
                    gen_config=genConfig,
                    sequence_start=True,
                    sequence_end=True,
                    stream_output=True):
                pass
            resQueue.put(QueueResponse(request, outputs))

    def _getSingleParameterSizeInBytes(self):
        return 2 