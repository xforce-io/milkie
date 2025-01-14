import traceback
from typing import Optional
import traceback
from clients.bird.task_alignment_searcher import TaskAlignmentSearcher
from clients.bird.config import Config
from clients.bird.logger import ERROR

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
            ERROR(f"Error in inference: {str(e)} {traceback.format_exc()}")
            raise