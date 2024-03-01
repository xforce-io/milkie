from llama_index import Response
from llama_index.schema import TextNode

from milkie.agent.base_agent import BaseAgent
from milkie.retrieval.retrieval import RetrievalModule


class RetrievalAgent(BaseAgent):

    def __init__(
            self,
            context,
            config):
        super().__init__(context, config)

        self.retrievalModule = RetrievalModule(
            retrievalConfig=self.config.retrievalConfig
        )

    def task(self, query) -> Response:
        self.context.setCurQuery(query)
        self.retrievalModule.retrieve(self.context)
        retrievalResult = self.context.retrievalResult
        response = Response()
        blocks = []
        for result in retrievalResult:
            blocks += [self.__getBlockFromNode(result.node)]
        response.metadata = {"blocks": blocks}
        return response

    def __getBlockFromNode(self, node :TextNode) ->str:
        block = node.get_text()
        curNode = node
        while True:
            nextNode = self.context.getGlobalMemory().getNextNode(curNode)
            if nextNode is None or len(block) > 2500:
                break

            block += nextNode.get_text()
            curNode = nextNode
        return block