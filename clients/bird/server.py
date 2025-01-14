from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uvicorn

from clients.bird.config import Config
from clients.bird.task_alignment_searcher import TaskAlignmentSearcher
from clients.bird.logger import ERROR, INFO, logger

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

class Server:
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config
        self.searcher = Searcher(config)
        self.app = FastAPI()
        
        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self.app.post("/v1/chat/completions")(self.chat_completion)
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # 提取查询
            query = None
            for msg in request.messages:
                if msg.role == "user":
                    query = msg.content
                    break
            
            if not query:
                query = ""
                
            # 执行搜索
            sql = self.searcher.inference(query)
            if sql is None:
                ERROR("Inference returned None")
                raise HTTPException(status_code=400, detail="Failed to generate valid SQL query")
            
            # 构建响应
            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time()*1000)}",
                object="chat.completion",
                created=int(time.time()),
                model="bird",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": sql 
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            ERROR(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    def run(self):
        INFO(f"Starting server on port {self.config.server.port}")
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.config.server.port
        )

def search(query: str, config: Optional[Config] = None) -> str:
    """
    执行text-to-sql搜索
    
    Args:
        query: 查询字符串
        config: 可选的配置对象
        
    Returns:
        str: 生成的SQL语句
    """
    try:
        searcher = TaskAlignmentSearcher(
            query=query,
            max_iters=config.search.max_iters if config else Config.load().search.max_iters,
            config=config
        )
        return searcher.inference()
    except Exception as e:
        ERROR(f"Search failed: {str(e)}")
        raise

def main():
    try:
        server = Server()
        server.run()
    except Exception as e:
        ERROR(f"Server failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    main()
