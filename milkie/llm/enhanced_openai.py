import logging
import re
from queue import Queue
from typing import Any, Sequence
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from llama_index_client import ChatMessage
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from milkie.cache.cache_kv import CacheKVMgr
from milkie.llm.enhanced_llm import EnhancedLLM, LLMApi, QueueRequest, QueueResponse
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
)

logger = logging.getLogger(__name__)

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
            if "chat" in model_name:
                self.model_name = "deepseek-chat"
            elif "coder" in model_name:
                self.model_name = "deepseek-coder"
            else:
                raise ValueError(f"Unknown model type: {model_name}")

        self.endpoint = endpoint
        self.api_key = api_key
        self._llm = LLMApi(
            context_window=context_window,
            model_name=model_name,
            client=OpenAI(api_key=api_key, base_url=endpoint))
        self._cacheMgr = CacheKVMgr("data/cache/", expireTimeByDay=7)

    def _chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        resps = self._completeBatchNoTokenizationAsync(
            prompts=[messages],
            numThreads=1,
            inference=EnhancedOpenAI._inference,
            **kwargs)
        return completion_response_to_chat_response(resps[0])

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
            
            messages = request.prompt
            messagesJson = []
            for message in messages:
                if message.role == "system":
                    messagesJson.append(ChatCompletionSystemMessageParam(
                        role=message.role,
                        content=message.content,
                    ))
                else:
                    content = re.sub(r'[\uD800-\uDFFF]', '', message.content)
                    messagesJson.append(ChatCompletionUserMessageParam(
                        role=message.role,
                        content=content,
                    ))

            cached = self._cacheMgr.getValue(
                modelName=self.model_name, 
                key=messagesJson)
            if cached:
                logger.debug("cache hit!")
                resQueue.put(QueueResponse(
                    requestId=request.requestId, 
                    chatCompletion=ChatCompletion.model_validate_json(cached["chatCompletion"]),
                    numTokens=cached["numTokens"]))
                return
            
            theArgs = ""
            try:
                theArgs = {
                    "model" : self.model_name,
                    "messages" : messagesJson,
                    "stream" : False,
                }
                if "tools" in genArgs:
                    theArgs["tools"] = genArgs["tools"]
                
                response = self._llm.getClient().chat.completions.create(**theArgs)
            except Exception as e:
                resQueue.put(QueueResponse(
                    requestId=request.requestId, 
                    chatCompletion="fail answer",
                    numTokens=1))
                logger.error(f"Failed to complete request[{request.prompt}]")
                return 
            
            resQueue.put(QueueResponse(
                requestId=request.requestId, 
                chatCompletion=response,
                numTokens=response.usage.completion_tokens))

            self._cacheMgr.setValue(
                modelName=self.model_name, 
                key=messagesJson, 
                value={
                    "chatCompletion" : response.model_dump_json(),
                    "numTokens" : response.usage.completion_tokens,
                })

    def _getSingleParameterSizeInBytes(self):
        return 0 