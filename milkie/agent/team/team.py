from llama_index.core import Response

from milkie.agent.base_agent import BaseAgent

class Team(BaseAgent):
    
    def task(self, query) -> Response:
        pass