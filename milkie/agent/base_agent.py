from abc import ABC, abstractmethod
from llama_index.core import Response

from milkie.context import Context

class BaseAgent(ABC):

    def __init__(
            self,
            context :Context=None,
            config :str=None) -> None:
        self.config = context.globalContext.globalConfig.agentsConfig.getConfig(config) if context else None
        self.setContext(context)

    def setContext(self, context :Context): 
        self.context = context

    @abstractmethod
    def task(self, query :str, **kwargs) -> Response:
        pass

    @abstractmethod
    def taskBatch(self, query :str, kwargs :list[dict]) -> list[Response]:
        pass
