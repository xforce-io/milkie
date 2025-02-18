from typing import Any, Dict, Optional, Sequence
from queue import Queue
from pydantic import PrivateAttr
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

    context_window: Optional[int] = Field(
        default=8192,
    )

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
    
    process: Any = PrivateAttr()
    engineArgs: Any = PrivateAttr()
    
    def __init__(
            self, 
            model_name: str,
            context_window: int,
            max_new_tokens :int,
            port :int,
            dtype :str,
            vllm_kwargs :Dict[str, Any]) -> None:
        super().__init__(model=model_name, max_new_tokens=max_new_tokens)

        self.model = model_name
        self.context_window = context_window
        self.port = port

        self.engineArgs = AsyncEngineArgs(
            model=model_name,
            max_model_len=context_window,
            dtype=dtype,
            **vllm_kwargs)

        self._startProcess()
        
    def __del__(self):
        if self.process is not None:
            self._endProcess()
        
    def class_name(cls) -> str:
        return "VLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self.context_window,)

    def _startProcess(self):
        import subprocess
        cmds = [
            'python', 
            '-m', 
            'milkie.llm.vllm_server']
        cmds += [f"--port", f"{str(self.port)}"]
        for key, value in vars(self.engineArgs).items():
            key = key.replace("_", "-")
            if key == "max-seq-len-to-capture":
                key = "max-seq_len-to-capture"
                
            if value is not None:
                if str(value) == "True":
                    cmds += [f"--{key}"]
                elif str(value) != "False":
                    cmds += [f"--{key}", f"{str(value)}"]


        self.process = subprocess.Popen(cmds)
        print(f"Started process with PID: [{self.process.pid}] command: [{cmds}]")
        
    def _endProcess(self):
        if self.process:
            self.process.terminate()
            print(f"Terminated process with PID: {self.process.pid}")
            self.process.wait()

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
            tensor_parallel_size: int,
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
            tensor_parallel_size=tensor_parallel_size,
            tokenizer_name=tokenizer_name, 
            model_name=model_name,
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
                "quantization" : None if quantMethod == QuantMethod.NONE else quantMethod.name,
                "tensor_parallel_size": self.tensor_parallel_size,})

    @torch.inference_mode()
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any):
        messages = self._llm._get_messages(prompt, **prompt_args)
        response = self._completion(messages)
        output = response.message.content or ""
        return (self._llm._parse_output(output), len(response.raw["model_output"]))

    def _getModel(self):
        return self._llm._engine

    def _completeBatch(
            self, 
            prompts: list[str], 
            **kwargs: Any
    ) -> list[CompletionResponse]:
        return self._completeBatchAsync(
            prompts=prompts, 
            numThreads=1,
            inference=EnhancedVLLM._inference,
            tokenIdExtractor=lambda output : output,
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
            
            data = {
                "prompt" : request.prompt,
                "prompt_token_ids": request.tokenized,
                **params
            }

            response = requests.post("http://0.0.0.0:%d/generate" % self.port, json=data)
            if response.status_code == 200:
                result = response.json()
                resQueue.put(QueueResponse(
                    requestId=request.requestId, 
                    chatCompletion=result["raw"]["model_output"]))
            else:
                raise ValueError("Failed to complete request, status code: %d" % response.status_code)

    def _getSingleParameterSizeInBytes(self):
        type_to_size = {
            "auto": 2,
        }

        dtype = self._llm.dtype
        size = type_to_size.get(dtype, None)
        if size is None:
            raise ValueError(f"Unsupported data type: {dtype}")
        return size
