import logging
from abc import abstractmethod

from llama_index.legacy.response.schema import Response
from llama_index.legacy.prompts import ChatPromptTemplate
from llama_index.legacy.llms.types import ChatMessage, MessageRole

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

    def task(self, query, **kwargs) -> Response:
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
            **kwargs)
        t1 = time.time()
        answer = response.response.replace("\n", "//")
        response.metadata["numTokens"] = numTokens
        logger.debug(f"prompt_agent query[{query}] answer[{answer}] cost[{t1-t0}]")
        return response

    def taskBatch(self, query, kwargs :list[dict]) -> list[Response]:
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
            argsList=kwargs)
        t1 = time.time()

        responses = []
        for result in resultBatch:
            response = Response(response=result[0], source_nodes=None, metadata={})
            #answer = response[0].replace("\n", "//")
            response.metadata["numTokens"] = result[1]
            responses += [response]
        logger.debug(f"prompt_agent query[{query}] batchSize[{len(resultBatch)}] cost[{t1-t0}]")
        return responses