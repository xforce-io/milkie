from typing import Any
import torch
from llama_index.legacy.core.llms.types import CompletionResponse
from llama_index.legacy.llms.huggingface import HuggingFaceLLM

from milkie.config.config import QuantMethod
from milkie.llm.enhanced_llm import EnhancedLLM

class EnhancedHFLLM(EnhancedLLM) :

    def __init__(
            self, 
            context_window: int, 
            max_new_tokens: int, 
            query_wrapper_prompt: str, 
            tokenizer_name: str, 
            model_name: str, 
            device: int,
            tokenizer_kwargs: dict, 
            model_kwargs: dict, 
            generate_kwargs: dict, 
            is_chat_model: bool, 
            system_prompt: str) -> None:
        tokenizer_kwargs["padding_side"] = "left"

        super().__init__(context_window, tokenizer_name, tokenizer_kwargs)

        compile = model_kwargs.pop("torch_compile", False)

        if device is not None:
            torch.cuda.set_device(device)

        from transformers import AutoModelForCausalLM
        from auto_gptq import AutoGPTQForCausalLM
        if EnhancedLLM.getQuantMethod(model_name) == QuantMethod.GPTQ:
            model = AutoGPTQForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

        self._llm = HuggingFaceLLM(
            context_window=context_window, 
            max_new_tokens=max_new_tokens, 
            query_wrapper_prompt=query_wrapper_prompt, 
            tokenizer=self._tokenizer,
            tokenizer_name=model_name,
            model_name=model_name, 
            model=model,
            model_kwargs=model_kwargs, 
            generate_kwargs=generate_kwargs, 
            is_chat_model=is_chat_model, 
            system_prompt=system_prompt)

        #refer suggestions from https://pytorch.org/blog/accelerating-generative-ai-2/
        if compile:
            self._llm._model = torch.compile(self._getModel(), mode="reduce-overhead", fullgraph=True)
        model_kwargs["torch_compile"] = compile

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

        inputs = self._llm._tokenizer(
            text=full_prompt, 
            return_tensors="pt", 
            padding=True)
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

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        inputs = self._llm._tokenizer(text=prompts, return_tensors="pt")
        for key in self._llm.tokenizer_outputs_to_remove:
            for input in inputs:
                if key in input:
                    inputs.pop(key, None)

        inputs = inputs.to(self._getModel().device)
        
        param = {**self._llm.generate_kwargs, **kwargs}
        tokensList = self._getModel().generate(
            **inputs,
            max_new_tokens=self._llm.max_new_tokens,
            stopping_criteria=self._llm._stopping_criteria,
            **param)

        completion_tokens = []
        for i in range(len(tokensList)):
            completion_tokens += [tokensList[i][len(inputs["input_ids"][i]):]]
        completion = self._llm._tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        completionResponses = []
        for i in range(len(tokensList)):
            completionResponses += [CompletionResponse(
                text=completion[i], 
                raw={"model_output": tokensList[i][len(inputs["input_ids"][i]):]})]
        return completionResponses

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