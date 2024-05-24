from typing import Any, Dict, Optional, Sequence
from queue import Queue
import requests
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse, 
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.bridge.pydantic import Field
from llama_index_client import BasePromptTemplate

from milkie.config.config import QuantMethod
from milkie.llm.enhanced_llm import EnhancedLLM, QueueRequest, QueueResponse

class VLLM(CustomLLM):

    model: Optional[str] = Field(description="The Vllm Model to use.")

    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate per output sequence.",
    )

    dtype: str = Field(
        default="auto",
        description="The data type for the model weights and activations.",
    )

    port : int = Field(
        default=8083,
        description="The port"
    )
    
    _process: Any = None
    
    def __init__(
            self, 
            model_name: str,
            context_window: int,
            max_new_tokens :int,
            port :int,
            dtype :str,
            vllm_kwargs :Dict[str, Any]) -> None:
        super().__init__(model=model_name, max_new_tokens=max_new_tokens)

        self.engineArgs = AsyncEngineArgs(
            model=model_name,
            max_model_len=context_window,
            dtype=dtype,
            **vllm_kwargs)

        self._port = port

        self._startProcess()
        
    def __del__(self):
        if self._process is not None:
            self._endProcess()
        
    def class_name(cls) -> str:
        return "VLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    def _startProcess(self):
        import subprocess
        cmds = [
            'python', 
            '-m', 
            'vllm.entrypoints.api_server']
        cmds += ["--port", str(self.port)]
        for key, value in self.engineArgs.items():
            cmds += [f"--{key}", str(value)]

        self._process = subprocess.Popen(cmds)
        self._pid = self.process.pid
        print(f"Started process with PID: [{self.pid}] command: [{cmds}]")
        
    def _endProcess(self):
        if self._process:
            self._process.terminate()
            print(f"Terminated process with PID: {self._pid}")
            self._process.wait()

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "max_tokens": self.max_new_tokens,
        }
        return {**base_kwargs}

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
            system_prompt: str,
            device :str,
            port :int,
            max_new_tokens: int,
            tokenizer_kwargs: dict):
        super().__init__(
            context_window=context_window, 
            concurrency=concurrency, 
            tokenizer_name=tokenizer_name, 
            system_prompt=system_prompt, 
            device=device, 
            port=port,
            tokenizer_kwargs=tokenizer_kwargs)
        
        quantMethod = EnhancedLLM.getQuantMethod(model_name)
        self._llm = VLLM(
            model_name=model_name,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            port=port,
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
        return self._llm._engine

    def _complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        return self._completeBatch([prompt], **kwargs)[0]

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> list[CompletionResponse]:
        return self._completeBatchAsync(
            prompts=prompts, 
            numThreads=1,
            inference=EnhancedVLLM._inference,
            tokenIdExtractor=lambda output : output.outputs[0].token_ids,
            **kwargs)

    def _inference(
            self, 
            reqQueue :Queue[QueueRequest], 
            resQueue :Queue[QueueResponse], 
            genArgs :dict,
            **kwargs :Any) -> Any:
        genArgs = genArgs if genArgs else {}
        params = {
            **self._llm._model_kwargs, 
            **EnhancedLLM.filterGenArgs(genArgs)}
        while not reqQueue.empty():
            request = reqQueue.get()
            if not request:
                break
            
            kwargs = kwargs if kwargs else {}
            data = {
                "prompt": request.prompt,
                **params
            }

            response = requests.post("http://0.0.0.0:%d" % self.port, json=data)
            if response.status_code == 200:
                result = response.json()
                resQueue.put(QueueResponse(
                    requestId=request.request_id, 
                    output=result["raw"]["model_output"]))
            else:
                raise ValueError("Failed to complete request")

    def _getSingleParameterSizeInBytes(self):
        type_to_size = {
            "auto": 2,
        }

        dtype = self._llm.dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size
