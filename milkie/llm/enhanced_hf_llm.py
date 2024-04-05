import torch
from typing import Any, Callable, Optional, Sequence
from llama_index import BasePromptTemplate
from llama_index.llms.types import ChatMessage
from llama_index.llms import HuggingFaceLLM

class EnhancedHFLLM(HuggingFaceLLM) :

    def __init__(
            self, 
            context_window: int, 
            max_new_tokens: int, 
            query_wrapper_prompt: str, 
            tokenizer_name: str, 
            model_name: str, 
            tokenizer_kwargs: dict, 
            model_kwargs: dict, 
            generate_kwargs: dict, 
            is_chat_model: bool, 
            system_prompt: str, 
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]]) -> None:
        compile = model_kwargs.pop("compile", False)

        super().__init__(
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
            self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=True)
        model_kwargs["compile"] = compile
    
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any) -> str:
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            response = self.chat(messages)
            output = response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt)
            output = response.text
        return (self._parse_output(output), len(response.raw["model_output"][0]))

    def getMem(self) -> float:
        return round(self._model.get_memory_footprint()/(1024*1024*1024), 2)

    def getNumParams(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    #get memory bandwidth utilization
    def getMBU(self, tokensPerSec :float, memBandwidth :float) -> float:
        return self.getNumParams() * self.__getSingleParameterSizeInBytes() * tokensPerSec / memBandwidth

    def __getSingleParameterSizeInBytes(self):
        type_to_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
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
