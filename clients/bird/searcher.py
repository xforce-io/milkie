from typing import Any, Optional
from clients.bird.base_searcher import Node, NodeType
from clients.bird.base_sql_searcher import BaseSqlSearcher
from clients.bird.task_alignment_searcher import TaskAlignmentSearcher
from milkie.sdk.agent_client import AgentClient
from milkie.sdk.config_server import ConfigServer
from clients.bird.config import Config
from clients.bird.database import Database
from clients.bird.logger import INFO, ERROR
from milkie.utils.data_utils import escape

class Searcher:
    def __init__(self, 
            config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config

    def inference(self, query: str) -> str:
        try:
            tree = TaskAlignmentSearcher(query, self.config.search.max_iters)
            return tree.inference()
        except Exception as e:
            ERROR(f"Error in inference: {str(e)}")
            raise