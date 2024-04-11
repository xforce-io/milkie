from typing import Any, Callable, Optional, Sequence
from vllm import SamplingParams
from llama_index.llms.vllm import Vllm
from llama_index_client import BasePromptTemplate, ChatMessage
from llama_index.legacy.core.llms.types import ChatMessage, CompletionResponse
import torch

from milkie.llm.enhanced_llm import EnhancedLLM

class EnhancedVLLM(EnhancedLLM):
    def __init__(self,
            model_name: str,
            max_new_tokens: int,
            message_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]]):
        self._llm = Vllm(
            model=model_name,
            tensor_parallel_size=1,
            max_new_tokens=max_new_tokens,
            vllm_kwargs={"swap_space":1, "gpu_memory_utilization":0.5},
            messages_to_prompt=message_to_prompt)

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
        params = {**self._model_kwargs, **kwargs}
        sampling_params = SamplingParams(**params)
        outputs = self._getModel().generate([prompt], sampling_params)
        return CompletionResponse(
            text=outputs[0].outputs[0].text,
            raw={"model_output": outputs[0].outputs[0].token_ids},)