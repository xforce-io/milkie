from typing import Any, Callable, Optional, Sequence
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
    
    def _getModel(self):
        return self._llm._client

    @torch.inference_mode()
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any) -> str:
        if self._llm.metadata.is_chat_model:
            messages = self._llm._get_messages(prompt, **prompt_args)
            response = self._chat(messages)
            output = response.message.content or ""
        else:
            raise NotImplementedError("predict not implemented for non-chat models")
        
        return (self._llm._parse_output(output), len(response.raw["model_output"][0]) - len(response.raw["model_input"][0]))


    def __complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._llm.complete(prompt, formatted=formatted, **kwargs)