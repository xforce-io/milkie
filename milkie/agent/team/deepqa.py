import logging
from llama_index.core import Response
from milkie.agent.qa_agent import QAAgent
from milkie.agent.team.team import Team
from milkie.context import Context

logger = logging.getLogger(__name__)

class DeepQA(Team):
   
    def __init__(
            self,
            context :Context,
            config :str) -> None:
         super().__init__(context, config)
         self.qaAgent = QAAgent(context, config)
      
    def task(self, query, **kwargs) -> Response:
        return self.qaAgent.task(query, **kwargs)

    def taskBatch(
            self, 
            prompt, 
            argsList :list, 
            **kwargs) -> Response:
        try:
            resps = []
            for args in argsList:
                resps.append(
                    self.qaAgent.task(prompt.format(**args), **kwargs))
            return resps
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error[{e}] in taskBatch")
            return None