from typing import Any
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

from llama_index.core.prompts.base import BasePromptTemplate
from llama_index_client import BasePromptTemplate
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.vllm import Vllm

from milkie.config.config import QuantMethod
from milkie.llm.enhanced_llm import EnhancedLLM

class EnhancedVLLM(EnhancedLLM):
    def __init__(self,
            context_window: int,
            concurrency: int,
            tokenizer_name: str,
            model_name: str,
            system_prompt: str,
            device :str,
            max_new_tokens: int,
            tokenizer_kwargs: dict):
        super().__init__(
            context_window=context_window, 
            concurrency=concurrency, 
            tokenizer_name=tokenizer_name, 
            system_prompt=system_prompt, 
            device=device, 
            tokenizer_kwargs=tokenizer_kwargs)
        
        quantMethod = EnhancedLLM.getQuantMethod(model_name)
        self._llm = Vllm(
            model=model_name,
            max_new_tokens=max_new_tokens,
            dtype="auto",
            vllm_kwargs={
                "gpu_memory_utilization":0.9, 
                "quantization" : None if quantMethod == QuantMethod.NONE else quantMethod.name})

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
        return self._completeBatchSync(prompts, **kwargs)

    def _completeBatchSync(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> list[CompletionResponse]:
        inputs = self._tokenizer(text=prompts, return_tensors="pt", padding=True)

        promptTokenIds = []
        for i in range(inputs["input_ids"].size(0)):
            promptTokenIds.append(EnhancedLLM._unpadTokenized(inputs, i).tolist())

        kwargs = kwargs if kwargs else {}
        params = {
            **self._llm._model_kwargs, 
            **EnhancedLLM.filterGenArgs(kwargs)}
        sampling_params = SamplingParams(**params)
        outputs = self._getModel().generate(
            prompt_token_ids=promptTokenIds, 
            sampling_params=sampling_params,
            use_tqdm=False)

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