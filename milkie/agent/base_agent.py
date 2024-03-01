from abc import ABC, abstractmethod
from llama_index import Response
from milkie.config.config import GlobalConfig
from milkie.context import Context

class BaseAgent(ABC):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        self.config = context.globalContext.globalConfig.agentsConfig.getConfig(config)
        self.context = context

    @abstractmethod
    def task(self, query) -> Response:
        pass