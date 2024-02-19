from llama_index import Response
from milkie.agent.qa_agent import QAAgent
from milkie.agent.team import Team
from milkie.config.config import GlobalConfig
from milkie.context import Context


class DeepQA(Team):
   
   def __init__(
            self,
            globalConfig :GlobalConfig,
            context :Context,
            config :str) -> None:
         super().__init__(globalConfig, context, config)
         self.qaAgent = QAAgent(globalConfig, context, config)
      
   def task(self, query) -> Response:
      return self.qaAgent.task(query)