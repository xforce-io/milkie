from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult, KeyResp
from milkie.context import Context
from milkie.functions.toolkits.base import BaseToolkit
from milkie.response import Response

class BaseBlock(ABC):

    def __init__(
            self,
            context :Context=None,
            config :str|GlobalConfig=None,
            toolkit :BaseToolkit=None,
            usePrevResult :bool=DefaultUsePrevResult) -> None:
        context = context if context else Context.createContext("config/global.yaml")
        self.setContext(context)

        if isinstance(config, str) or config is None:
            self.config = context.globalContext.globalConfig.agentsConfig.getConfig(config) if config else context.globalContext.globalConfig
        else:
            self.config = config

        self.toolkit = toolkit
        self.usePrevResult = usePrevResult

    def setContext(self, context :Context): 
        self.context = context

    @abstractmethod
    def execute(self, query :str, args :dict, prevBlock :BaseBlock=None, **kwargs) -> Response:
        pass

    def executeBatch(self, query :str, argsList :list[dict], **kwargs) -> list[Response]:
        return [self.execute(query, args, **kwargs) for args in argsList]

    def compile(self):
        pass

    def updateFromPrevBlock(self, prevBlock :BaseBlock, args :dict={}):
        if prevBlock:
            self.updateVarDict(prevBlock.getVarDict())
        self.updateVarDict(args)

    def getVarDict(self):
        return self.context.varDict

    def setVarDict(self, key: str, val):
        self.context.varDict[key] = val

    def setResp(self, key: str, val: Any):
        self.context.varDict[KeyResp][key] = val

    def updateVarDict(self, newDict: dict):
        self.context.varDict.update(newDict)

    def clearVarDict(self):
        self.context.varDict.clear()
        self.context.varDict[KeyResp] = {}
