from typing import Any, Callable, Optional, Sequence

from transformers import AutoTokenizer

from llama_index.core.prompts.base import BasePromptTemplate
from vllm import SamplingParams
from llama_index.legacy.llms.vllm import Vllm
from llama_index_client import BasePromptTemplate, ChatMessage
from llama_index.legacy.core.llms.types import ChatMessage, CompletionResponse
import torch

from milkie.llm.enhanced_llm import EnhancedLLM

class EnhancedVLLM(EnhancedLLM):
    def __init__(self,
            model_name: str,
            device :str,
            max_new_tokens: int,
            message_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]],
            tokenizer_kwargs: dict,):
        if device is not None:
            torch.cuda.set_device(device)

        self._llm = Vllm(
            model=model_name,
            tensor_parallel_size=1,
            max_new_tokens=max_new_tokens,
            vllm_kwargs={"gpu_memory_utilization":0.75},
            messages_to_prompt=message_to_prompt,
            dtype="auto",)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, kwargs=tokenizer_kwargs)
        self._llm._client.set_tokenizer(self._tokenizer)
        
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
        return self._llm._client

    def _complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._llm._model_kwargs, **kwargs}
        sampling_params = SamplingParams(**params)
        outputs = self._getModel().generate([prompt], sampling_params)
        return CompletionResponse(
            text=outputs[0].outputs[0].text,
            raw={"model_output": outputs[0].outputs[0].token_ids},)

    def _completeBatch(
        self, prompts: list[str], formatted: bool = False, **kwargs: Any
    ) -> list[CompletionResponse]:
        kwargs = kwargs if kwargs else {}
        params = {**self._llm._model_kwargs, **kwargs}
        sampling_params = SamplingParams(repetition_penalty=1.2, **params)
        outputs = self._getModel().generate(prompts, sampling_params)

        result = []
        for output in outputs:
            result += [CompletionResponse(
                text=output.outputs[0].text,
                raw={"model_output": output.outputs[0].token_ids},)]
        return result

    def _getSingleParameterSizeInBytes(self):
        type_to_size = {
            "auto": 2,
        }

        dtype = self._llm.dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size