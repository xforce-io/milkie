import random
from typing import Any, Optional, Sequence
from milkie.llm.enhanced_llm import EnhancedLLM
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig

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
    _client: Any = PrivateAttr()
    
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
        turboMind = TurboMind.from_pretrained(model_name, engineConfig)
        self._client = turboMind.create_instance()

    def modelInst(self):
        return self._client

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

class EnhancedLmDeploy(EnhancedLLM):
    def __init__(
            self, 
            context_window: int, 
            tokenizer_name: str, 
            model_name: str,
            device: str,
            max_new_tokens: int,
            tokenizer_kwargs: dict) -> None:
        tokenizer_kwargs["padding_side"] = "left"

        super().__init__(context_window, tokenizer_name, device, tokenizer_kwargs)

        self._llm = LMDeploy(
                model_name, 
                context_window)
        self.device = device

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        inputs = self._tokenizer(text=prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        engineOutputs = self.modelInst().batched_infer(
            session_ids=[random.randint(0, 1000000) for _ in range(len(prompts))],
            token_ids=inputs,
        )

        completionTokens = []
        for i in range(len(engineOutputs)):
            completionTokens += [engineOutputs[i][len(inputs["input_ids"][i]):]]
        completion = self._tokenizer.batch_decode(completionTokens, skip_special_tokens=True)

        completionResponses = []
        for i in range(len(engineOutputs)):
            completionResponses += [CompletionResponse(
                text=completion[i], 
                raw={"model_output": engineOutputs[i][len(inputs["input_ids"][i]):]})]
        return completionResponses