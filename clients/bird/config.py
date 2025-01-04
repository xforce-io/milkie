import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServerConfig:
    port: int

@dataclass
class AgentConfig:
    addr: str
    name: str

@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

@dataclass
class SearchConfig:
    max_thoughts: int
    min_sqls: int
    max_sqls: int
    max_iters: int

@dataclass
class ModelConfig:
    thought_model: str  # 用于生成思考的模型
    sql_model: str      # 用于生成SQL的模型

@dataclass
class Config:
    server: ServerConfig
    agent: AgentConfig
    database: DatabaseConfig
    search: SearchConfig
    model: ModelConfig
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> 'Config':
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'bird.yaml')
            
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return Config(
            server=ServerConfig(**data['server']),
            agent=AgentConfig(**data['agent']),
            database=DatabaseConfig(**data['database']),
            search=SearchConfig(**data['search']),
            model=ModelConfig(**data['model'])
        )
