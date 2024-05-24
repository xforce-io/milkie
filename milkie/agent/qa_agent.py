from llama_index.core import Response

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
            context :Context,
            config :str):
        super().__init__(context, config)

        if self.config.memoryConfig and self.config.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                context.globalContext.settings,
                self.config.memoryConfig,
                self.config.indexConfig)
        else:
            self.memoryWithIndex = context.getGlobalContext().memoryWithIndex

        self.groundingModule = GroundingModule()
        self.retrievalModule = RetrievalModule(
            globalConfig=context.globalContext.globalConfig,
            retrievalConfig=self.config.retrievalConfig,
            memoryWithIndex=self.memoryWithIndex)
        self.reasoningModule = ReasoningModule()
        self.decisionModule = DecisionModule(
            engine=self.retrievalModule.engine)
        self.actionModule = ActionModule()

    def task(self, query, **kwargs) -> Response:
        self.context.setCurQuery(query)
        self.processRound(self.context)
        while not self._end():
            self.processRound(self.context)
        return self.context.decisionResult

    def taskBatch(self, query: str, kwargs: list[dict]) -> list[Response]:
        raise NotImplementedError("QAAgent does not support taskBatch")

    def processRound(self, context, **kwargs):
        self.__retrieval(context, **kwargs)
        self.__reasoning(context, **kwargs)
        self.__decision(context, **kwargs)
    
    def _end(self) -> bool:
        return True

    def __retrieval(self, context, **kwargs):
        return self.retrievalModule.retrieve(context, **kwargs)

    def __reasoning(self, context, **kwargs):
        return self.reasoningModule.reason(context, **kwargs)

    def __decision(self, context, **kwargs):
        return self.decisionModule.decide(context, **kwargs)

if __name__ == "__main__":
    globalConfig = GlobalConfig("config/global.yaml")
    context = Context(globalConfig)
    agent = QAAgent(globalConfig, context, "qa")
    agent.task("你好")