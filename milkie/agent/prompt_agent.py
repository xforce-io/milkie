from abc import abstractmethod

from llama_index import PromptTemplate, Response
from milkie.agent.base_agent import BaseAgent
from milkie.config.config import GlobalConfig, PromptAgentConfig
from milkie.context import Context
from milkie.prompt.prompt import GLoader


class PromptAgent(BaseAgent):

    def __init__(
            self,
            globalConfig :GlobalConfig,
            context :Context,
            config :str) -> None:
        self.__init__(globalConfig, context, config)

        self.prompt = GLoader.load(config)

    def task(self, query) -> Response:
        response = Response()
        response.response = self.context.settings.llm.stream(PromptTemplate(query))
        return response