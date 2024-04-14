import torch
from typing import Any, Callable, Optional, Sequence
from llama_index.legacy.core.llms.types import ChatMessage, CompletionResponse
from llama_index.legacy.llms.huggingface import HuggingFaceLLM

from milkie.llm.enhanced_llm import EnhancedLLM

class EnhancedHFLLM(EnhancedLLM) :

    def __init__(
            self, 
            context_window: int, 
            max_new_tokens: int, 
            query_wrapper_prompt: str, 
            tokenizer_name: str, 
            model_name: str, 
            device: str,
            tokenizer_kwargs: dict, 
            model_kwargs: dict, 
            generate_kwargs: dict, 
            is_chat_model: bool, 
            system_prompt: str, 
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]]) -> None:
        compile = model_kwargs.pop("torch_compile", False)

        self._llm = HuggingFaceLLM(
            context_window=context_window, 
            max_new_tokens=max_new_tokens, 
            query_wrapper_prompt=query_wrapper_prompt, 
            tokenizer_name=tokenizer_name, 
            model_name=model_name, 
            tokenizer_kwargs=tokenizer_kwargs, 
            model_kwargs=model_kwargs, 
            generate_kwargs=generate_kwargs, 
            is_chat_model=is_chat_model, 
            system_prompt=system_prompt, 
            messages_to_prompt=messages_to_prompt)

        #refer suggestions from https://pytorch.org/blog/accelerating-generative-ai-2/
        if compile:
            self._llm._model = torch.compile(self._getModel(), mode="reduce-overhead", fullgraph=True)
        model_kwargs["torch_compile"] = compile

        if device:
            self._llm._model.to(device)

    def getMem(self) -> float:
        return round(self._getModel().get_memory_footprint()/(1024*1024*1024), 2)

    def getNumParams(self) -> int:
        return sum(p.numel() for p in self._getModel().parameters())

    def _getModel(self):
        return self._llm._model
    
    def _complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if self._llm.query_wrapper_prompt:
                full_prompt = self._llm.query_wrapper_prompt.format(query_str=prompt)
            if self._llm.system_prompt:
                full_prompt = f"{self._llm.system_prompt} {full_prompt}"

        inputs = self._llm._tokenizer(text=full_prompt, return_tensors="pt")
        inputs = inputs.to(self._getModel().device)

        for key in self._llm.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        tokens = self._getModel().generate(
            **inputs,
            max_new_tokens=self._llm.max_new_tokens,
            stopping_criteria=self._llm._stopping_criteria,
            **self._llm.generate_kwargs,
        )
        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        completion = self._llm._tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return CompletionResponse(
            text=completion, 
            raw={"model_output": tokens[0][len(inputs["input_ids"][0]):]})

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