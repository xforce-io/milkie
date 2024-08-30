import logging
from llama_index.core import Response
from milkie.agent.qa_block import QABlock
from milkie.agent.agents.base_agent import BaseAgent
from milkie.context import Context

logger = logging.getLogger(__name__)

class DeepQA(BaseAgent):
   
    def __init__(
            self,
            context :Context,
            config :str) -> None:
         super().__init__(context, config)
         self.qaAgent = QABlock(context, config)
      
    def execute(self, query, **kwargs) -> Response:
        return self.qaAgent.execute(query, **kwargs)

    def executeBatch(
            self, 
            prompt, 
            argsList :list, 
            **kwargs) -> Response:
        try:
            resps = []
            for args in argsList:
                resps.append(
                    self.qaAgent.execute(prompt.format(**args), **kwargs))
            return resps
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error[{e}] in taskBatch")
            return None