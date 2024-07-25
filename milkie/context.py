from typing import List
from llama_index.legacy.response.schema import Response
from llama_index.core.base.response.schema import NodeWithScore

from milkie.global_context import GlobalContext
from milkie.utils.req_tracer import ReqTracer

class Context:
    def __init__(
            self, 
            globalContext :GlobalContext) -> None:
        self.globalContext = globalContext
        self.reqTrace = ReqTracer()
            
        self.curQuery :str = None
        self.retrievalResult :List[NodeWithScore] = None
        self.decisionResult :Response = None
        self.engine = None
        self.instructions = []
        
    def getGlobalContext(self):
        return self.globalContext

    def getGlobalMemory(self):
        memoryWithIndex = self.globalContext.memoryWithIndex
        return memoryWithIndex.memory if memoryWithIndex else None
    
    def setCurQuery(self, query):
        self.curQuery = query

    def getCurQuery(self):
        return self.curQuery

    def getCurInstruction(self):
        return None if len(self.instructions) == 0 else self.instructions[-1]

    def setRetrievalResult(self, retrievalResult :List[NodeWithScore]):
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult :Response):
        self.decisionResult = decisionResult
