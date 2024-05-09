from typing import Any, Dict, Optional, Sequence
from queue import Queue
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.legacy.llms.custom import CustomLLM
from llama_index_client import BasePromptTemplate
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse, 
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.legacy.bridge.pydantic import Field, PrivateAttr

from milkie.config.config import QuantMethod
from milkie.llm.enhanced_llm import EnhancedLLM, QueueRequest, QueueResponse

class VLLM(CustomLLM):

    model: Optional[str] = Field(description="The Vllm Model to use.")

    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )
    
    _engine: Any = PrivateAttr()
    
    def __init__(
            self, 
            model_name: str,
            context_window: int,
            max_new_tokens :int,
            vllm_kwargs :Dict[str, Any]) -> None:
        super().__init__(model=model_name, max_new_tokens=max_new_tokens)

        engineArgs = EngineArgs(
            model=model_name,
            max_model_len=context_window,
            **vllm_kwargs)
        self._engine = LLMEngine.from_engine_args(engineArgs)
        
    def class_name(cls) -> str:
        return "VLLM"

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "max_tokens": self.max_new_tokens,
        }
        return {**base_kwargs}

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


class EnhancedVLLM(EnhancedLLM):
    def __init__(self,
            context_window: int,
            concurrency: int,
            tokenizer_name: str,
            model_name: str,
            device :str,
            max_new_tokens: int,
            tokenizer_kwargs: dict):
        super().__init__(context_window, concurrency, tokenizer_name, device, tokenizer_kwargs)
        
        self._llm = VLLM(
            context_window=context_window,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            vllm_kwargs={
                "gpu_memory_utilization":0.9, 
                "quantization" : None if EnhancedLLM.getQuantMethod(model_name) == QuantMethod.NONE else "gptq",
                "enable_prefix_caching" :True})

    @torch.inference_mode()
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any):
        messages = self._llm._get_messages(prompt, **prompt_args)
        response = self._chat(messages)
        output = response.message.content or ""
        return (self._llm._parse_output(output), len(response.raw["model_output"]))

    def _getModel(self):
        return self._llm.engine

    def _complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._llm._model_kwargs, **kwargs}
        sampling_params = SamplingParams(
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            **params)
        outputs = self._getModel().generate([prompt], sampling_params)
        return CompletionResponse(
            text=outputs[0].outputs[0].text,
            raw={"model_output": outputs[0].outputs[0].token_ids},)

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> list[CompletionResponse]:
        return self._completeBatchAsync(
            prompts=prompts, 
            inference=EnhancedVLLM._inference,
            tokenIdExtractor=lambda output : output.outputs[0].token_ids,
            **kwargs)

    def _completeBatchSync(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> list[CompletionResponse]:
        inputs = self._tokenizer(text=prompts, return_tensors="pt", padding=True)

        promptTokenIds = []
        for i in range(inputs["input_ids"].size(0)):
            promptTokenIds.append(inputs["input_ids"][i][:inputs["attention_mask"][i].sum(dim=0)])

        kwargs = kwargs if kwargs else {}
        params = {
            **self._llm._model_kwargs, 
            **EnhancedLLM.filterGenArgs(kwargs)}
        sampling_params = SamplingParams(**params)
        outputs = self._getModel().generate(
            prompt_token_ids=promptTokenIds, 
            sampling_params=sampling_params)

        result = []
        for output in outputs:
            result += [CompletionResponse(
                text=output.outputs[0].text,
                raw={"model_output": output.outputs[0].token_ids},)]
        return result

    def _inference(
            self, 
            reqQueue :Queue[QueueRequest], 
            resQueue :Queue[QueueResponse], 
            genArgs :dict,
            **kwargs :Any) -> Any:
        for request in iter(reqQueue.get, None):
            genArgs = genArgs if genArgs else {}
            params = {
                **self._llm._model_kwargs, 
                **EnhancedLLM.filterGenArgs(genArgs)}
            samplingParams = SamplingParams(**params)
            self._llm.engine.add_request(
                    prompt=request.prompt,
                    prompt_token_ids=request.tokenized, 
                    sampling_params=samplingParams, 
                    request_id=request.requestId)

            outputs = []
            while self._llm.engine.has_unfinished_requests():
                stepOutputs = self._llm.engine.step()
                for requestOutput in stepOutputs:
                    if requestOutput.finished:
                        outputs.append(requestOutput)

            for output in outputs:
                resQueue.put(QueueResponse(
                    requestId=request.requestId, 
                    output=output))

    def _getSingleParameterSizeInBytes(self):
        type_to_size = {
            "auto": 2,
        }

        dtype = self._llm.dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size