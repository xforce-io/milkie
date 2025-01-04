import logging
import re
from queue import Queue
from typing import Any, Sequence, Generator
from llama_index_client import ChatMessage
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from llama_index.core.base.llms.types import ChatResponseGen, CompletionResponse, CompletionResponseGen, ChatResponse
from milkie.cache.cache_kv import CacheKVMgr
from milkie.llm.enhanced_llm import EnhancedLLM, LLMApi, QueueRequest, QueueResponse
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)

import uuid
import time
import json

logger = logging.getLogger(__name__)

class EnhancedOpenAI(EnhancedLLM):
    # 类级别常量
    MAX_RETRIES = 1  # 最大重试次数
    
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

        self.endpoint = endpoint
        self.api_key = api_key
        self._llm = LLMApi(
            context_window=context_window,
            model_name=model_name,
            client=OpenAI(api_key=api_key, base_url=endpoint))
        self._cacheMgr = CacheKVMgr("data/cache/", expireTimeByDay=7)

    def _fail(self, messages :Sequence[ChatMessage], **kwargs: Any):
        self._cacheMgr.removeValue(
            modelName=self.model_name, 
            key=self._createMessagesJson(messages))

    def _createMessagesJson(self, messages: Sequence[ChatMessage]) -> list:
        """创建 OpenAI API 所需的消息格式"""
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
        return messagesJson

    def _createApiArgs(self, messagesJson: list, stream: bool = False, **kwargs: Any) -> dict:
        """创建 API 调用参数"""
        theArgs = {
            "model": self.model_name,
            "messages": messagesJson,
            "stream": stream,
        }
        if "tools" in kwargs:
            theArgs["tools"] = kwargs["tools"]
        return theArgs

    def _createCacheResponse(self, content: str, toolCalls: list) -> dict:
        """创建用于缓存的响应格式"""
        # 将 toolCalls 转换为可序列化的字典格式
        serializableToolCalls = []
        if toolCalls:
            for toolCall in toolCalls:
                if toolCall:
                    serializableToolCalls.append({
                        "id": toolCall.id,
                        "type": toolCall.type,
                        "function": {
                            "name": toolCall.function.name,
                            "arguments": toolCall.function.arguments
                        }
                    })

        return {
            "id": "cached_" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": serializableToolCalls
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": len(content.split()),
                "prompt_tokens": 0,
                "total_tokens": len(content.split())
            }
        }

    def _setCacheValue(
            self, 
            messagesJson: list, 
            content: str, 
            toolCalls: list | None,
            numTokens: int) -> None:
        """设置缓存值"""
        cacheResponse = self._createCacheResponse(content, toolCalls)
        self._cacheMgr.setValue(
            modelName=self.model_name,
            key=messagesJson,
            value={
                "chatCompletion": json.dumps(cacheResponse),
                "numTokens": numTokens
            }
        )

    def _completion(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messagesJson = self._createMessagesJson(messages)
        cached = self._cacheMgr.getValue(modelName=self.model_name, key=messagesJson)
        
        if cached and ("no_cache" not in kwargs or kwargs["no_cache"] is False):
            logger.debug("cache hit!")
            chatCompletion = ChatCompletion.model_validate_json(cached["chatCompletion"])
            numTokens = cached["numTokens"]
        else:
            retry_count = 0
            while retry_count <= self.MAX_RETRIES:
                try:
                    self._addDisturbance(messagesJson, **kwargs)
                    theArgs = self._createApiArgs(messagesJson, stream=False, **kwargs)
                    response = self._llm.getClient().chat.completions.create(**theArgs)
                    chatCompletion = response
                    numTokens = response.usage.completion_tokens
                    self._setCacheValue(
                        messagesJson=messagesJson, 
                        content=chatCompletion.choices[0].message.content, 
                        toolCalls=chatCompletion.choices[0].message.tool_calls,
                        numTokens=numTokens)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count > self.MAX_RETRIES:
                        return self._handleApiError(e)
                    logger.warning(f"Retry {retry_count}/{self.MAX_RETRIES} after error: {str(e)}")

        return completion_response_to_chat_response(CompletionResponse(
            text=chatCompletion.choices[0].message.content or "",
            raw={
                "model_output": None,
                "num_tokens": numTokens,
                "chat_completion": chatCompletion
            }
        ))

    def _stream(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        return stream_completion_response_to_chat_response(self._streamImpl(messages, **kwargs))

    def _streamImpl(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        messagesJson = self._createMessagesJson(messages)
        cached = self._cacheMgr.getValue(modelName=self.model_name, key=messagesJson)
        
        if cached and ("no_cache" not in kwargs or kwargs["no_cache"] is False):
            logger.debug("cache hit in stream!")
            chatCompletion = ChatCompletion.model_validate_json(cached["chatCompletion"])
            yield from self._simulateStream(
                content=chatCompletion.choices[0].message.content, 
                toolCalls=chatCompletion.choices[0].message.tool_calls)
            return
        
        retry_count = 0
        while retry_count <= self.MAX_RETRIES:
            try:
                self._addDisturbance(messagesJson, **kwargs)
                theArgs = self._createApiArgs(messagesJson, stream=True, **kwargs)
                stream = self._llm.getClient().chat.completions.create(**theArgs)
                fullContent = []
                funcName = None
                funcArgs = []
                
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        fullContent.append(delta.content)
                        yield CompletionResponse(text=delta.content, raw=delta)
                    elif delta.tool_calls:
                        if delta.tool_calls[0].function.name is not None:
                            funcName = delta.tool_calls[0].function.name
                        if delta.tool_calls[0].function.arguments is not None:
                            funcArgs.append(delta.tool_calls[0].function.arguments)
                        yield CompletionResponse(text="", raw=delta)

                content = "".join(fullContent)
                arguments = "".join(funcArgs) if funcArgs else ""
                self._setCacheValue(
                    messagesJson=messagesJson, 
                    content=content, 
                    toolCalls=[
                        ChatCompletionMessageToolCall(
                            id=str(uuid.uuid4()),
                            type="function",
                            function=Function(
                                name=funcName, 
                                arguments=arguments
                            )
                        )
                    ] if funcName is not None else None,
                    numTokens=len(content.split()))
                break
            except Exception as e:
                retry_count += 1
                if retry_count > self.MAX_RETRIES:
                    self._handleApiError(e, isStream=True)
                logger.warning(f"Retry {retry_count}/{self.MAX_RETRIES} after error: {str(e)}")

    def _simulateStream(
            self, 
            content: str, 
            toolCalls: list | None, 
            chunk_size: int = 1024) -> Generator[CompletionResponse, None, None]:
        """模拟流式输出"""
        if toolCalls is None or len(toolCalls) == 0:
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                yield CompletionResponse(text=chunk, raw=ChoiceDelta(
                    content=chunk
                ))
        else:
            yield CompletionResponse(text="", raw=ChoiceDelta(
                content=None,
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=str(uuid.uuid4()),
                        type="function", 
                        function=ChoiceDeltaToolCallFunction( 
                            name=toolCalls[0].function.name,   
                            arguments=toolCalls[0].function.arguments
                        )
                    )
                ]
            ))

    def _addDisturbance(self, messagesJson: list, **kwargs) -> None:
        if "no_cache" in kwargs and kwargs["no_cache"] is True:
            import random
            import string
            messagesJson[0]["content"] = random.choice(string.ascii_letters) + messagesJson[0]["content"]

    #deprecated
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
            messagesJson = self._createMessagesJson(messages)

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
            
            theArgs = self._createApiArgs(messagesJson, stream=False)
            try:
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

    def _handleApiError(self, e: Exception, isStream: bool = False) -> Any:
        """统一处理 API 错误"""
        error_msg = f"Failed to {'stream' if isStream else 'complete'} request: {e} model: {self.model_name}"
        logger.error(error_msg)
        if isStream:
            raise RuntimeError(error_msg)
        return ChatResponse(message=ChatMessage(role="assistant", content="Failed to generate response"))
