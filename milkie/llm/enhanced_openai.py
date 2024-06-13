from queue import Queue
from typing import Any
from llama_index.core.llms.llm import LLM
from openai import OpenAI
from llama_index.core.base.llms.types import CompletionResponse
from milkie.llm.enhanced_llm import EnhancedLLM, LLMApi, QueueRequest, QueueResponse

class EnhancedOpenAI(EnhancedLLM):
    def __init__(self, 
            model_name :str,
            system_prompt :str,
            endpoint :str,
            api_key :str,
            context_window :int,
            concurrency :int,
            tensor_parallel_size :int,
            tokenizer_name :str,
            device :str,
            port :int,
            tokenizer_kwargs :dict):
        super().__init__(
            context_window=context_window,
            concurrency=concurrency,
            tensor_parallel_size=tensor_parallel_size,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            system_prompt=system_prompt,
            device=device,
            port=port,
            tokenizer_kwargs=tokenizer_kwargs)

        if model_name.startswith("deepseek"):
            self.model_name = "deepseek-chat"
        self.endpoint = endpoint
        self.api_key = api_key
        self._llm = LLMApi(OpenAI(api_key=api_key, base_url=endpoint))

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any) -> list[CompletionResponse]:
        return self._completeBatchAsync(
            prompts=prompts, 
            numThreads=1,
            inference=EnhancedOpenAI._inference,
            tokenIdExtractor=lambda output : output,
            **kwargs)

    def _inference(
            self, 
            reqQueue :Queue[QueueRequest], 
            resQueue :Queue[QueueResponse], 
            genArgs :dict,
            **kwargs :Any) -> Any:
        while not reqQueue.empty():
            request = reqQueue.get()
            if not request:
                break
            
            messages = []
            if self._systemPrompt:
                messages.append({"role": "system", "content": self._systemPrompt})
            messages.append({"role": "user", "content": request.prompt})
            try:
                response = self._llm.getClient().chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False
                )
            except Exception as e:
                raise ValueError("Failed to complete request, status code: %d" % response.status_code)
            
            resQueue.put(QueueResponse(
                requestId=request.requestId, 
                output=response.choices[0].message.content))