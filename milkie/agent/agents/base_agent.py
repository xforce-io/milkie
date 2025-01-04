from milkie.agent.base_block import BaseBlock
from milkie.response import Response

class BaseAgent(BaseBlock):
    
    def execute(self, query) -> Response:
        pass