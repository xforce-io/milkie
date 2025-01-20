import traceback
from typing import Optional
import traceback
from clients.bird.database import Database
from clients.bird.text2sql_searcher import Text2SqlSearcher
from clients.bird.config import Config
from clients.bird.logger import ERROR
from milkie.sdk.agent_client import AgentClient
from milkie.sdk.config_server import ConfigServer

class Searcher:
    def __init__(self, 
            config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config
        self._client = AgentClient(ConfigServer(config.agent.addr))
        self._db = Database(config.database, self._client)

    def inference(self, query: str) -> str:
        try:
            tree = Text2SqlSearcher(
                client=self._client,
                database=self._db,
                query=query,
                max_iters=self.config.search.max_iters
            )
            return tree.inference()
        except Exception as e:
            ERROR(f"Error in inference: {str(e)} {traceback.format_exc()}")
            raise