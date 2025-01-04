import requests
from typing import Optional, Dict, Any, Union

from milkie.sdk.config_server import ConfigServer

class AgentClient:
    def __init__(self, config: Union[ConfigServer, str]):
        """
        初始化 AgentClient
        
        Args:
            config: 可以是 ConfigServer 实例或服务器地址字符串
        """
        if isinstance(config, str):
            self.config = ConfigServer(config)
        else:
            self.config = config
    
    def execute(self, code: str, agent_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 agent
        
        Args:
            code: 要执行的代码
            agent_name: agent 的名称
            args: 可选的参数字典
        
        Returns:
            执行结果字符串
        
        Raises:
            Exception: 当请求失败或服务器返回错误时抛出
        """
        if args is None:
            args = {}
            
        url = f"{self.config.getAddr()}/v1/agent"
        
        data = {
            "agent": agent_name,
            "code": code,
            "args": args
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # 检查 HTTP 错误
            
            result = response.json()
            
            # 检查服务器返回的错误
            if result.get("errno", 0) != 0:
                raise Exception(result.get("errmsg", "Unknown error"))
                
            return result.get("resp", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        except ValueError as e:
            raise Exception(f"Failed to parse response: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
