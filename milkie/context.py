from __future__ import annotations
from queue import Queue
import copy
from datetime import datetime
import time
from typing import Dict, Generator, List, Optional, Any
from llama_index.core.base.response.schema import NodeWithScore

from milkie.agent.memory.history import History
from milkie.agent.memory.memory import Memory
from milkie.agent.query_structure import QueryStructure, parseQuery
from milkie.agent.exec_graph import ExecGraph
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.trace import stdout
from milkie.utils.req_tracer import ReqTracer

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
        self.memory = Memory(
            self.globalContext.globalConfig.memoryConfig,
            self.globalContext)
        self.respQueue = Queue()
        self.graphQueue = Queue()
        self.execGraph = ExecGraph()
        self.vm = self.globalContext.vm
        self.history = History()

    def getGlobalContext(self):
        return self.globalContext

    def getEnv(self):
        return self.globalContext.env

    def getGlobalDocset(self):
        docsetWithIndex = self.globalContext.docsetWithIndex
        return docsetWithIndex.memory if docsetWithIndex else None
    
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

    def getHistory(self) -> History:
        return self.history

    def setRetrievalResult(self, retrievalResult: List[NodeWithScore]) -> None:
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult: Response) -> None:
        self.decisionResult = decisionResult

    def setSystemPrompt(self, systemPrompt: str) -> None:
        self.history.setSystemPrompt(systemPrompt)

    def getVarDict(self) -> VarDict:
        return self.varDict

    def historyAddUserPrompt(self, userPrompt: str) -> None:
        self.history.addUserPrompt(userPrompt)
        
    def historyAddAssistantPrompt(self, assistantPrompt: str) -> None:
        self.history.addAssistantPrompt(assistantPrompt)

    def getRespStream(self) -> Generator[str, None, None]:
        while True:
            item = self.respQueue.get()
            if item is None:
                break
            yield item

    def getGraphStream(self) -> Generator[str, None, None]:
        while True:
            try:
                # 每次发送最新的执行图数据
                execGraphData = self.execGraph.dump()
                print(f"发送执行图数据 (大小: {len(execGraphData)}字节)", flush=True)
                yield execGraphData
            except Exception as e:
                print(f"生成执行图数据时出错: {str(e)}", flush=True)
            
            # 检查是否有更新信号
            if not self.graphQueue.empty():
                signal = self.graphQueue.get()
                if signal is None:  # 结束信号
                    print("收到执行图流结束信号", flush=True)
                    break
            
            # 短暂等待后再次发送更新
            time.sleep(0.5)

    def triggerGraphUpdate(self) -> None:
        """触发执行图更新"""
        self.graphQueue.put("update")

    def copy(self) -> Context:
        newContext = Context(self.globalContext)
        newContext.query = self.query
        newContext.retrievalResult = self.retrievalResult
        newContext.decisionResult = self.decisionResult
        newContext.instructions = self.instructions
        newContext.varDict = self.varDict.copy()
        return newContext

    def closeStream(self) -> None:
        self.respQueue.put(None)
        self.graphQueue.put(None)

    @staticmethod
    def create(configPath: Optional[str] = None) -> Context:
        """创建新的上下文实例"""
        if Context.globalContext:
            return Context(Context.globalContext)

        Context.globalContext = GlobalContext.create(configPath)
        return Context(Context.globalContext)

    def genResp(self, info, **kwargs):
        if "end" in kwargs:
            self.respQueue.put(str(info) + kwargs["end"])
            stdout(str(info), info=True, flush=True, end=kwargs["end"])
        else:
            self.respQueue.put(str(info) + "\n")
            stdout(str(info), info=True, flush=True)

    def genExecGraph(self, **kwargs):
        self.respQueue.put(self.execGraph.dump())
