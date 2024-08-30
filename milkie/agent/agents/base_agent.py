from llama_index.core import Response

from milkie.agent.base_block import BaseBlock

class BaseAgent(BaseBlock):
    
    def execute(self, query) -> Response:
        pass