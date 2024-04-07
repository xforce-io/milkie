import logging
from abc import abstractmethod

from llama_index.legacy.prompts import ChatPromptTemplate, Response
from llama_index.llms.types import ChatMessage, MessageRole

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.prompt.prompt import Loader

logger = logging.getLogger(__name__)

class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        super.__init__(context, config)

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