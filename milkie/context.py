from __future__ import annotations
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any
from llama_index.core.base.response.schema import NodeWithScore
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.agent.query_structure import QueryStructure, parseQuery
from milkie.agent.exec_graph import ExecGraph
from milkie.response import Response
from milkie.utils.req_tracer import ReqTracer
from milkie.global_context import GlobalContext

class VarDict:
    """变量字典类，管理全局和局部变量"""
    def __init__(self):
        self.varDict = {
            "_date": datetime.now().strftime("%Y-%m-%d")
        }
        self.globalDict: Dict[str, Any] = {
            "_date": datetime.now().strftime("%Y-%m-%d")
        }
        self.localDict: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """获取变量值，优先从局部字典获取"""
        return self.localDict.get(key) or self.globalDict.get(key)

    def getAllDict(self) -> dict:
        """获取合并后的全局和局部字典"""
        return {**self.globalDict, **self.localDict}

    def getGlobalDict(self) -> dict:
        return self.globalDict

    def getLocalDict(self) -> dict:
        return self.localDict
    
    def setGlobal(self, key: str, value: Any) -> None:
        assert key is not None
            
        if value is None:
            self.globalDict.pop(key, None)
        else:
            self.globalDict[key] = value

    def setLocal(self, key: str, value: Any) -> None:
        assert key is not None
            
        if value is None:
            self.localDict.pop(key, None)
        else:
            self.localDict[key] = value

    def setResp(self, key: str, value: Any) -> None:
        """设置响应变量"""
        if value is None:
            self.globalDict.pop(key, None)
        else:
            self.globalDict[key] = value

    def update(self, newDict: VarDict) -> None:
        """更新字典内容"""
        self.globalDict.update(newDict.globalDict)
        self.localDict.update(newDict.localDict)

    def updateFromDict(self, newDict :dict) -> None:
        self.globalDict.update(newDict)

    def updateFromDictLocal(self, newDict :dict) -> None:
        self.localDict.update(newDict)

    def copy(self) -> VarDict:
        result = copy.deepcopy(self)
        return result

    def remove(self, key: str) -> None:
        self.localDict.pop(key, None)
        self.globalDict.pop(key, None)

    def clear(self) -> None:
        """清空所有字典"""
        self.localDict.clear()
        self.globalDict.clear()

    def clearLocal(self, params :List[str]) -> None:
        """只清空局部字典"""
        for param in params:
            self.localDict.pop(param, None)

    def __str__(self):
        return str(self.getAllDict())

    def __iter__(self):
        return iter(self.globalDict)

class History:
    """对话历史管理类"""
    def __init__(self):
        self.systemPrompt: Optional[str] = None
        self.history: List[ChatMessage] = []
        self.resetUse()

    def resetUse(self) -> None:
        self.activate = True

    def use(self) -> bool:
        """历史记录在单个agent运行中只能使用一次"""
        value = self.activate
        self.activate = False
        return value

    def setSystemPrompt(self, systemPrompt: str) -> None:
        self.systemPrompt = systemPrompt

    def addHistoryUserPrompt(self, userPrompt: str) -> None:
        self.history.append(ChatMessage(content=userPrompt, role=MessageRole.USER))

    def addHistoryAssistantPrompt(self, assistantPrompt: str) -> None:
        self.history.append(ChatMessage(content=assistantPrompt, role=MessageRole.ASSISTANT))

    def getDialogue(self) -> List[ChatMessage]:
        """获取完整对话历史，包括系统提示"""
        if self.systemPrompt:
            return [ChatMessage(content=self.systemPrompt, role=MessageRole.SYSTEM)] + self.history
        return self.history

    def getDialogueStr(self) -> str:
        """获取对话历史的字符串形式"""
        return "\n".join(msg.content for msg in self.history)

    def copy(self) -> History:
        return copy.deepcopy(self)

class Context:
    """上下文管理类"""
    globalContext = None
    
    def __init__(self, globalContext) -> None:
        self.globalContext = globalContext
        self.reqTrace = ReqTracer()
        self.query: Optional[QueryStructure] = None
        self.retrievalResult: Optional[List[NodeWithScore]] = None
        self.decisionResult: Optional[Response] = None
        self.instructions: List[Any] = []
        self.varDict = VarDict()
        self.history = History()
        self.execGraph = ExecGraph()

    def getGlobalContext(self):
        return self.globalContext

    def getEnv(self):
        return self.globalContext.env

    def getGlobalMemory(self):
        memoryWithIndex = self.globalContext.memoryWithIndex
        return memoryWithIndex.memory if memoryWithIndex else None
    
    def getExecGraph(self) -> ExecGraph:
        return self.execGraph

    def setQuery(self, query: str) -> None:
        self.query = parseQuery(query)
        self.execGraph.start(query)

    def getQuery(self) -> Optional[QueryStructure]:
        return self.query

    def getQueryStr(self) -> Optional[str]:
        return self.query.query if self.query else None

    def getInstructions(self) -> List[Any]:
        return self.instructions

    def setRetrievalResult(self, retrievalResult: List[NodeWithScore]) -> None:
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult: Response) -> None:
        self.decisionResult = decisionResult

    def setSystemPrompt(self, systemPrompt: str) -> None:
        self.history.setSystemPrompt(systemPrompt)

    def getVarDict(self) -> VarDict:
        return self.varDict

    def addHistoryUserPrompt(self, userPrompt: str) -> None:
        self.history.addHistoryUserPrompt(userPrompt)
        
    def addHistoryAssistantPrompt(self, assistantPrompt: str) -> None:
        self.history.addHistoryAssistantPrompt(assistantPrompt)

    def getHistory(self) -> History:
        return self.history

    def copy(self) -> Context:
        newContext = Context(self.globalContext)
        newContext.query = self.query
        newContext.retrievalResult = self.retrievalResult
        newContext.decisionResult = self.decisionResult
        newContext.instructions = self.instructions
        newContext.varDict = self.varDict.copy()
        return newContext

    @staticmethod
    def create(configPath: Optional[str] = None) -> Context:
        """创建新的上下文实例"""
        if Context.globalContext:
            return Context(Context.globalContext)

        Context.globalContext = GlobalContext.create(configPath)
        return Context(Context.globalContext)