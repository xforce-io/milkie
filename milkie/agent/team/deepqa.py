from llama_index.core import Response
from milkie.agent.qa_agent import QAAgent
from milkie.agent.team.team import Team
from milkie.config.config import GlobalConfig
from milkie.context import Context


class DeepQA(Team):
   
    def __init__(
            self,
            context :Context,
            config :str) -> None:
         super().__init__(context, config)
         self.qaAgent = QAAgent(context, config)
      
    def task(self, query, **kwargs) -> Response:
        return self.qaAgent.task(query, kwargs)

    def taskBatch(self, queries: list[str]) -> list[Response]:
        raise NotImplementedError("DeepQA does not support taskBatch")