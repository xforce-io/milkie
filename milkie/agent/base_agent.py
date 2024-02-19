from abc import ABC, abstractmethod
from llama_index import Response
from milkie.config.config import GlobalConfig
from milkie.context import Context

class BaseAgent(ABC):

    def __init__(
            self,
            globalConfig :GlobalConfig,
            context :Context,
            config :str) -> None:
        self.globalConfig = globalConfig
        self.config = globalConfig.agentsConfig.getConfig(config)
        self.context = context

    @abstractmethod
    def task(self, query) -> Response:
        pass