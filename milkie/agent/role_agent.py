from llama_index.core import Response
from milkie.agent.base_agent import BaseAgent
from milkie.context import Context


class RoleAgent(BaseAgent):
    
    def __init__(
            self, 
            context: Context, 
            config: str,
            role: str,
            goal: str,
            backstory: str,
            tools :list) -> None:
        super().__init__(context, config)

        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
    
    def execute(self, query: str, **kwargs) -> Response:
        return super().execute(query, **kwargs)