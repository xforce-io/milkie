from typing import List
from llama_index.core.base.response.schema import NodeWithScore

from milkie.agent.query_structure import QueryStructure, parseQuery
from milkie.config.config import GlobalConfig
from milkie.config.constant import KeyResp
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.response import Response
from milkie.utils.req_tracer import ReqTracer

class Context:

    globalContext :GlobalContext = None
    
    def __init__(
            self, 
            globalContext :GlobalContext) -> None:
        self.globalContext = globalContext
        self.reqTrace = ReqTracer()
            
        self.curQuery :QueryStructure = None
        self.retrievalResult :List[NodeWithScore] = None
        self.decisionResult :Response = None
        self.engine = None
        self.instructions = []
        self.varDict = {KeyResp: {}}
        
    def getGlobalContext(self):
        return self.globalContext

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

        Context.globalContext = GlobalContext.create(configPath)
        return Context(Context.globalContext)