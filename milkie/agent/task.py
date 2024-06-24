from llama_index.core import Response

from milkie.agent.role_agent import RoleAgent

class Task:
    def __init__(
            self, 
            description: str,
            expectedOutput: str) -> None:
        pass

    def execute(
            self, 
            agent: RoleAgent, 
            **kwargs) -> Response:
        pass