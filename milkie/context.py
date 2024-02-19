from llama_index import Response
from milkie.settings import Settings

class Context:
    def __init__(
            self, 
            settings :Settings) -> None:
        self.settings = settings
            
        self.curQuery :str = None
        self.retrievalResult = None
        self.decisionResult :Response = None
        self.engine = None
        self.instructions = []
        
    def setCurQuery(self, query):
        self.curQuery = query

    def getCurQuery(self):
        return self.curQuery

    def getCurInstruction(self):
        return None if len(self.instructions) == 0 else self.instructions[-1]

    def setRetrievalResult(self, retrievalResult):
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult :Response):
        self.decisionResult = decisionResult
