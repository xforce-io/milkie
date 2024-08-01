from abc import ABC, abstractmethod
from llama_index.core import Response

from milkie.context import Context

class BaseAgent(ABC):

    def __init__(
            self,
            context :Context=None,
            config :str=None) -> None:
        self.config = context.globalContext.globalConfig.agentsConfig.getConfig(config) if context else None
        self.logger = context.globalContext.logger
        self.setContext(context)

    def setContext(self, context :Context): 
        self.context = context

    @abstractmethod
    def execute(self, query :str, args :dict, **kwargs) -> Response:
        pass

    def executeBatch(self, query :str, argsList :list[dict], **kwargs) -> list[Response]:
        return [self.execute(query, args, **kwargs) for args in argsList]
