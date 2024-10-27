from __future__ import annotations
from typing import List
from llama_index.core.base.response.schema import NodeWithScore

from milkie.agent.query_structure import QueryStructure, parseQuery
from milkie.config.constant import KeyResp
from milkie.response import Response
from milkie.utils.req_tracer import ReqTracer

class VarDict:
    def __init__(self):
        self.globalDict = {KeyResp: {}}
        self.localDict = {}

    def get(self, key :str):
        resp = self.localDict.get(key)
        if resp:
            return resp
        return self.globalDict.get(key)

    def getAllDict(self):
        return {**self.globalDict, **self.localDict}

    def getGlobalDict(self):
        return self.globalDict

    def getLocalDict(self):
        return self.localDict
    
    def setGlobal(self, key :str, value):
        if value is None:
            self.globalDict.pop(key)
        else:
            self.globalDict[key] = value

    def setLocal(self, key :str, value):
        if value is None:
            self.localDict.pop(key)
        else:
            self.localDict[key] = value

    def setResp(self, key :str, value):
        if value is None:
            self.globalDict[KeyResp].pop(key)
        else:
            self.globalDict[KeyResp][key] = value

    def update(self, newDict :VarDict):
        self.globalDict.update(newDict.globalDict)
        self.localDict.update(newDict.localDict)

    def updateFromDict(self, newDict :dict):
        self.globalDict.update(newDict)

    def clear(self):
        self.localDict.clear()
        self.globalDict.clear()
        self.globalDict[KeyResp] = {}

    def clearLocal(self):
        self.localDict.clear()


class Context:

    globalContext = None
    
    def __init__(
            self, 
            globalContext) -> None:
        self.globalContext = globalContext
        self.reqTrace = ReqTracer()
            
        self.curQuery :QueryStructure = None
        self.retrievalResult :List[NodeWithScore] = None
        self.decisionResult :Response = None
        self.engine = None
        self.instructions = []
        self.varDict = VarDict()
        
    def getGlobalContext(self):
        return self.globalContext

    def getEnv(self):
        return self.globalContext.env

    def getGlobalMemory(self):
        memoryWithIndex = self.globalContext.memoryWithIndex
        return memoryWithIndex.memory if memoryWithIndex else None
    
    def setCurQuery(self, query :str):
        self.curQuery = parseQuery(query)

    def getCurQuery(self) -> QueryStructure:
        return self.curQuery

    def getCurInstruction(self):
        return None if len(self.instructions) == 0 else self.instructions[-1]

    def setRetrievalResult(self, retrievalResult :List[NodeWithScore]):
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult :Response):
        self.decisionResult = decisionResult

    @staticmethod
    def create(configPath :str = None):
        if Context.globalContext:
            return Context(Context.globalContext)

        from milkie.global_context import GlobalContext
        Context.globalContext = GlobalContext.create(configPath)
        return Context(Context.globalContext)