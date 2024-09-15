from milkie.action import ActionModule
from milkie.agent.base_block import BaseBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.decision.decision import DecisionModule
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.reasoning import ReasoningModule
from milkie.response import Response
from milkie.retrieval.retrieval import RetrievalModule


class QABlock(BaseBlock):

    def __init__(
            self, 
            context :Context = None,
            config :str = None):
        super().__init__(context, config)

        if self.config.memoryConfig and self.config.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                self.context.globalContext.settings,
                self.config.memoryConfig,
                self.config.indexConfig,
                self.context.globalContext.serviceContext)
        else:
            self.memoryWithIndex = context.getGlobalContext().memoryWithIndex

        self.retrievalModule = RetrievalModule(
            globalConfig=self.context.globalContext.globalConfig,
            retrievalConfig=self.config.retrievalConfig,
            memoryWithIndex=self.memoryWithIndex,
            context=self.context)
        self.reasoningModule = ReasoningModule()
        self.decisionModule = DecisionModule()
        self.actionModule = ActionModule()

    def execute(self, query, args :dict, **kwargs) -> Response:
        self.context.setCurQuery(query)
        self.processRound(self.context)
        while not self._end():
            self.processRound(self.context)
        return self.context.decisionResult

    def executeBatch(self, query: str, argsList :list[dict], **kwargs) -> list[Response]:
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
        self.decisionModule.setEngine(self.retrievalModule.engine)
        return self.decisionModule.decide(context, **kwargs)

if __name__ == "__main__":
    agent = QABlock(config="qa")
    agent.execute("你好", args=None)