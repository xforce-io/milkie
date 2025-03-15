import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Generator, Union
import uvicorn
import json

from milkie.runtime.engine import Engine
from milkie.response import Response
from milkie.log import ERROR

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class AgentCompletionRequest(BaseModel):
    agent: str
    code: str
    args: dict
    
class AgentCompletionResponse(BaseModel):
    errno: int
    errmsg: str
    resp: str

class Server:
    def __init__(self, engine: Engine, agent_name: str):
        self.engine = engine
        self.agent_name = agent_name
        self.app = FastAPI()
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.post("/v1/chat/completions")(self.chat_completion)
        self.app.post("/v1/agent")(self.execute_agent)
       
    async def chat_completion(self, request: ChatCompletionRequest):
        try:
            # 提取 system prompt 和用户查询
            system_prompt = None
            query = None
            
            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    query = msg.content
            
            # 准备参数
            args = {
                "query": query,
                "system_prompt": system_prompt,
                "stream": request.stream
            }
            
            # 生成响应ID
            response_id = f"chatcmpl-{int(time.time()*1000)}"
            
            # 调用 engine
            response = self.engine.run(agent=self.agent_name, query=query, args=args)
            
            # 处理流式响应
            if request.stream and isinstance(response, Response) and response.respGen:
                return StreamingResponse(
                    self.create_stream_response(response_id, response.respGen),
                    media_type="text/event-stream"
                )
            
            # 处理普通响应
            return ChatCompletionResponse(
                id=response_id,
                object="chat.completion",
                created=int(time.time()),
                model=self.agent_name,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response)
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def execute_agent(self, request: AgentCompletionRequest):
        if request.agent != self.agent_name:
            ERROR(f"No suitable agent found for request: {request.agent}")
            raise HTTPException(status_code=400, detail="No suitable agent found")
        
        try:
            response = self.engine.executeAgent(
                agentName=request.agent, 
                code=request.code, 
                args=request.args)
            return AgentCompletionResponse(
                errno=0,
                errmsg="",
                resp=str(response)
            )
        except Exception as e:
            return AgentCompletionResponse(
                errno=1,
                errmsg=str(e),
                resp=""
            )

    def create_stream_response(self, response_id: str, content_gen: Generator) -> Generator:
        """创建流式响应"""
        try:
            for chunk in content_gen:
                # 确保 chunk 是字符串
                chunk_str = str(chunk)
                if not chunk_str:
                    continue
                    
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.agent_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": chunk_str
                        },
                        "finish_reason": None
                    }]
                }
                # 添加调试日志
                print(f"Sending chunk: {chunk_str}", flush=True)
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
            # 发送结束标记
            end_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.agent_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Stream error: {str(e)}", flush=True)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port) 