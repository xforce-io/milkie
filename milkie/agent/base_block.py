from abc import ABC, abstractmethod

from llama_index.core import Response
from milkie.config.config import GlobalConfig
from milkie.context import Context


class BaseBlock(ABC):

    def __init__(
            self,
            context :Context=None,
            config :str|GlobalConfig=None) -> None:
        context = context if context else Context.createContext("config/global.yaml")
        self.setContext(context)

        if isinstance(config, str) or config is None:
            self.config = context.globalContext.globalConfig.agentsConfig.getConfig(config) if config else context.globalContext.globalConfig
        else:
            self.config = config

    def setContext(self, context :Context): 
        self.context = context

    @abstractmethod
    def execute(self, query :str, args :dict, **kwargs) -> Response:
        pass

    def executeBatch(self, query :str, argsList :list[dict], **kwargs) -> list[Response]:
        return [self.execute(query, args, **kwargs) for args in argsList]
