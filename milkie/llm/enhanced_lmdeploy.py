import random
from typing import Any
from milkie.llm.enhanced_llm import EnhancedLLM
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig
from llama_index.legacy.core.llms.types import CompletionResponse

class EnhancedLmDeploy(EnhancedLLM):
    def __init__(
            self, 
            context_window: int, 
            tokenizer_name: str, 
            model_name: str,
            device: str,
            max_new_tokens: int,
            tokenizer_kwargs: dict) -> None:
        super().__init__(context_window, tokenizer_name, device, tokenizer_kwargs)

        engineConfig = TurbomindEngineConfig(
            cache_max_entry_count=0.8,
            cache_block_seq_len=64,
            model_format="hf",
            session_len=self.context_window,
            tp=1)
        self._llm = TurboMind.from_pretrained(model_name, engineConfig)
        self._modelInst = self._llm.create_instance()

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        inputs = self._tokenizer(text=prompts, return_tensors="pt")
        inputs = inputs.to(self._getModel().device)
        
        engineOutputs = self._modelInst.batched_infer(
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