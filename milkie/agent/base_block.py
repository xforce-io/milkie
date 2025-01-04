from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult
from milkie.context import Context, VarDict
from milkie.functions.toolkits.toolkit import EmptyToolkit, Toolkit
from milkie.response import Response

class BaseBlock(ABC):

    def __init__(
            self, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Optional[Toolkit] = None,
            usePrevResult=DefaultUsePrevResult,
            repoFuncs=None):
        self.context = context or Context.create()
        self.config = config if config else self.context.globalContext.globalConfig
        self.toolkit = toolkit
        if self.toolkit is None:
            self.toolkit = EmptyToolkit(self.context.globalContext)
            
        self.usePrevResult = usePrevResult
        self.repoFuncs = repoFuncs
        self.isCompiled = False

    def setContext(self, context :Context): 
        self.context = context

    @abstractmethod
    def execute(
            self, 
            context: Context,
            query :str, 
            args :dict, 
            prevBlock :BaseBlock=None, 
            **kwargs) -> Response:
        if context:
            self.setContext(context)
        self.updateFromPrevBlock(prevBlock, args)

    def executeBatch(
            self, 
            context: Context,
            query :str, 
            argsList :list[dict], 
            **kwargs) -> list[Response]:
        return [self.execute(context, query, args, **kwargs) for args in argsList]

    @abstractmethod
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
        self.context.varDict.setLocal(key, val)

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

    def clearVarDictLocal(self, params :List[str]):
        self.context.varDict.clearLocal(params)