import json
import time
import uvicorn
import threading
import logging
from typing import List, Dict, Any, Optional, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from milkie.runtime.engine import Engine
from milkie.response import Response
from milkie.log import ERROR

logger = logging.getLogger(__name__)

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

class AgentInfo(BaseModel):
    agent: str
    description: str
    args: dict

class AgentInfosRequest(BaseModel):
    agent: str

class AgentInfosResponse(BaseModel):
    errno: int
    errmsg: str
    resp: List[AgentInfo]

class AgentQueryRequest(BaseModel):
    agent: str
    query: str
    args: dict

class AgentQueryResponse(BaseModel):
    errno: int
    errmsg: str
    resp: str

class Server:
    def __init__(self, engine: Engine, agent_name: str):
        self.engine = engine
        self.agent_name = agent_name if agent_name else ""
        self.app = FastAPI()
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有源，生产环境中建议限制为前端域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"]
        )
        
        self.app.post("/v1/chat/completions")(self.chat_completion)
        self.app.post("/v1/agent/exec")(self.execute_agent)
        self.app.post("/v1/agent/infos")(self.get_agent_infos)
        self.app.get("/v1/agent/stream")(self.stream_agent)
       
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
            response = self.engine.run(agent=self.agent_name, args=args)
            
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
            ERROR(logger, f"No suitable agent found for request: {request.agent}")
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
    
    async def get_agent_infos(self, request: AgentInfosRequest):
        """获取代理信息的接口"""
        if request.agent == "":
            agentInfos = self.engine.getAllAgents()
            resp = []
            for agentName, agent in agentInfos.items():
                resp.append(AgentInfo(
                    agent=agentName,
                    description=agent.desc,
                    args={}
                ))
            return AgentInfosResponse(
                errno=0,
                errmsg="",
                resp=resp)
        else:
            agent = self.engine.getAgent(request.agent)
            if agent is None:
                ERROR(logger, f"Agent {request.agent} not found")
                raise HTTPException(status_code=400, detail="Agent not found")
            return AgentInfosResponse(
                errno=0,
                errmsg="",
                resp=AgentInfo(
                    agent=agent.name,
                    description=agent.description,
                    args={}
                )
            )

    async def stream_agent(self, request: Request):
        """流式执行代理查询的接口"""
        try:
            # 从请求参数中获取agent和query
            params = request.query_params
            agent_name = params.get("agent")
            query = params.get("query")
            graph = params.get("graph")
            
            if not agent_name or not query:
                raise HTTPException(status_code=400, detail="Missing agent or query parameter")
            
            print(f"接收到流式查询请求: agent={agent_name}, query={query}", flush=True)
            
            # 检查代理是否存在
            agent = self.engine.getAgent(agent_name)
            if agent is None:
                ERROR(logger, f"Agent {agent_name} not found")
                raise HTTPException(status_code=400, detail="Agent not found")
            
            # 准备参数，添加stream=True标志
            args = {
                "query": query,
                "stream": True
            }
            
            response_id = f"agentstream-{int(time.time()*1000)}"
            context = self.engine.createContext(args)

            def run_engine():
                self.engine.run(
                    context=context,
                    agent=agent.name,
                    args=args
                )
                context.closeStream()

            thread = threading.Thread(target=run_engine)
            thread.start()
            
            if graph:
                return StreamingResponse(
                    self.create_agent_stream_response(response_id, context.getGraphStream()),
                    media_type="text/event-stream"
                )
            else:
                return StreamingResponse(
                    self.create_agent_stream_response(response_id, context.getRespStream()),
                    media_type="text/event-stream"
                )
                
        except Exception as e:
            ERROR(logger, f"Stream agent error: {str(e)}")
            print(f"流式查询错误: {str(e)}", flush=True)
            # 返回错误响应
            error_data = {
                "errno": 1,
                "errmsg": str(e),
                "resp": ""
            }
            async def error_response():
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "event: complete\ndata: {}\n\n"
            return StreamingResponse(
                error_response(),
                media_type="text/event-stream"
            )
    
    def create_agent_stream_response(self, response_id: str, content_gen: Generator) -> Generator:
        """创建代理流式响应"""
        try:
            for chunk in content_gen:
                # 确保chunk是字符串
                chunk_str = str(chunk)
                if not chunk_str:
                    continue
                
                # 判断是否为执行图数据
                is_graph_data = False
                try:
                    # 尝试解析为JSON，看是否包含nodes字段
                    parsed_data = json.loads(chunk_str)
                    if "nodes" in parsed_data:
                        is_graph_data = True
                        # 直接发送执行图数据
                        chunk_data = {
                            "errno": 0,
                            "errmsg": "",
                            "nodes": parsed_data["nodes"]
                        }
                        print(f"发送执行图数据，节点数: {len(parsed_data['nodes'])}", flush=True)
                    else:
                        # 普通响应数据
                        chunk_data = {
                            "errno": 0,
                            "errmsg": "",
                            "resp": chunk_str
                        }
                except json.JSONDecodeError:
                    # 不是有效的JSON，当作普通响应文本处理
                    chunk_data = {
                        "errno": 0,
                        "errmsg": "",
                        "resp": chunk_str
                    }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            yield "event: complete\ndata: {}\n\n"
            
        except Exception as e:
            print(f"Agent stream error: {str(e)}", flush=True)
            error_data = {
                "errno": 1,
                "errmsg": str(e),
                "resp": ""
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "event: complete\ndata: {}\n\n"

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port) 