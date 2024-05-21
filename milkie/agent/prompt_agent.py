import logging

from llama_index.core import Response
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.prompt.prompt import Loader

logger = logging.getLogger(__name__)

class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        super().__init__(context, config)

        self.prompt = Loader.load(config) if config else None

    def task(self, query :str, argsList :list[dict], **kwargs) -> Response:
        response = Response(response="", source_nodes=None, metadata={})
        chatPromptTmpl = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=self.prompt if self.prompt else query,
                    role=MessageRole.USER)
            ]
        )

        import time
        t0 = time.time()
        response.response, numTokens = self.context.globalContext.settings.llm.predict(
            prompt=chatPromptTmpl,
            **argsList[0])
        t1 = time.time()
        answer = response.response.replace("\n", "//")
        response.metadata["numTokens"] = numTokens
        logger.debug(f"prompt_agent query[{query}] answer[{answer}] cost[{t1-t0}]").replace("\n", "//")
        return response

    def taskBatch(self, query :str, argsList :list[dict], **kwargs) -> list[Response]:
        chatPromptTmpl = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=self.prompt if self.prompt else query,
                    role=MessageRole.USER)
            ]
        )

        import time
        t0 = time.time()
        resultBatch = self.context.globalContext.settings.llm.predictBatch(
            prompt=chatPromptTmpl,
            argsList=argsList,
            **kwargs)
        t1 = time.time()

        responses = []
        for result in resultBatch:
            response = Response(response=result[0], source_nodes=None, metadata={})
            response.metadata["numTokens"] = result[1]
            responses += [response]
        logger.debug(f"prompt_agent query[{query}] batchSize[{len(resultBatch)}] cost[{t1-t0}]".replace("\n", "//"))
        return responses