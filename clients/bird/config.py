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
    max_sqls: int

@dataclass
class Config:
    server: ServerConfig
    agent: AgentConfig
    database: DatabaseConfig
    search: SearchConfig
    
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
            search=SearchConfig(**data['search'])
        )
