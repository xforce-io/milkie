from llama_index.core import Response
from llama_index.core.schema import TextNode

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.retrieval.retrieval import RetrievalModule


class RetrievalAgent(BaseAgent):

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

        self.retrievalModule = RetrievalModule(
            globalConfig=context.globalContext.globalConfig,
            retrievalConfig=self.config.retrievalConfig,
            memoryWithIndex=self.memoryWithIndex,
        )

    def execute(self, query) -> Response:
        self.context.setCurQuery(query)
        self.retrievalModule.retrieve(self.context)
        retrievalResult = self.context.retrievalResult
        response = Response(response="", source_nodes=None)
        blocks = []
        for result in retrievalResult:
            blocks += [self.__getBlockFromNode(result.node)]
        response.metadata = {"blocks": blocks}
        return response

    def __getBlockFromNode(self, node :TextNode) ->str:
        return node.get_text()