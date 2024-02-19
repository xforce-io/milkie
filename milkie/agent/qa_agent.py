from llama_index import Response
from milkie.action import ActionModule
from milkie.agent.base_agent import BaseAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.decision.decision import DecisionModule
from milkie.grounding import GroundingModule
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.reasoning import ReasoningModule
from milkie.retrieval.retrieval import RetrievalModule

class QAAgent(BaseAgent):

    def __init__(
            self, 
            globalConfig :GlobalConfig,
            context :Context,
            config :str):
        super().__init__(globalConfig, context, config)

        self.memoryWithIndex = MemoryWithIndex(
            context.settings,
            self.config.memoryConfig,
            self.config.indexConfig)

        self.grounding_module = GroundingModule()
        self.retrieval_module = RetrievalModule(
            qaAgentConfig=self.config,
            memoryWithIndex=self.memoryWithIndex)
        self.reasoning_module = ReasoningModule()
        self.decision_module = DecisionModule(
            engine=self.retrieval_module.engine)
        self.action_module = ActionModule()

    def task(self, query) -> Response:
        self.context.setCurQuery(query)
        self.processRound(self.context)
        while not self._end():
            self.processRound(self.context)
        return self.context.decisionResult

    def processRound(self, context):
        self.__retrieval(context)
        self.__reasoning(context)
        self.__decision(context)
    
    def _end(self) -> bool:
        return True

    def __retrieval(self, context):
        return self.retrieval_module.retrieve(context)

    def __reasoning(self, context):
        return self.reasoning_module.reason(context)

    def __decision(self, context):
        return self.decision_module.decide(context)

if __name__ == "__main__":
    globalConfig = GlobalConfig("config/global.yaml")
    context = Context(globalConfig)
    agent = QAAgent(globalConfig, context, "qa")
    agent.task("你好")