from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult
from milkie.context import Context, VarDict
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.response import Response

class BaseBlock(ABC):

    def __init__(
            self, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            repoFuncs=None):
        self.context = context or Context.create()
        self.config = config
        self.toolkit = toolkit
        self.usePrevResult = usePrevResult
        self.repoFuncs = repoFuncs

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
        self.updateVarDictFromDict(args)

    def getEnv(self):
        return self.context.getEnv()

    def getVarDict(self) -> VarDict:
        return self.context.varDict

    def setVarDictGlobal(self, key: str, val):
        self.context.varDict.setGlobal(key, val)

    def setVarDictLocal(self, key: str, val):
        self.context.varDict.setLocal(key, val)

    def getVarDictValue(self, key: str):
        return self.context.varDict.get(key)

    def setResp(self, key: str, val: Any):
        self.context.varDict.setResp(key, val)

    def updateVarDict(self, newDict: VarDict):
        self.context.varDict.update(newDict)

    def updateVarDictFromDict(self, newDict: dict):
        self.context.varDict.updateFromDict(newDict)

    def clearVarDict(self):
        self.context.varDict.clear()

    def clearVarDictLocal(self):
        self.context.varDict.clearLocal()
