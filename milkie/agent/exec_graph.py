from __future__ import annotations
from enum import Enum
from typing import List
import uuid
import json

from milkie.log import ERROR

class ExecNodeType(Enum):
    ROOT = 0
    COMMON = 1
    SEQUENCE = 2
    CALL = 3
    FOR = 4

class ExecNodeLabel(Enum):
    ROOT = 0
    AGENT = 1
    BLOCK = 2
    LLM = 3
    SKILL = 4
    TOOL = 5

class ExecNode:

    def __init__(
            self, 
            execGraph: ExecGraph,
            type: ExecNodeType,
            label: ExecNodeLabel = None,
            instructionId: str = None):
        self.execGraph = execGraph
        self.type = type
        self.label = label
        self.id = self.createId()
        self.parentNode :ExecNode = None
        self.instructionId :str = instructionId

        self.execGraph.addNode(self)

    def toDict(self) -> dict:
        """将节点转换为字典表示"""
        base = {
            "id": self.id,
            "type": self.type.name,
        }

        if self.label:
            base["label"] = self.label.name

        if self.instructionId:
            base["instruction"] = self.instructionId
        return base

    def createId(self):
        return f"{self.type.name}_{uuid.uuid4()}"

    def setInstructionId(self, instructionId: str):
        self.instructionId = instructionId

    def getId(self):
        return self.id

    def getInstructionId(self):
        return self.instructionId
        
    def castTo(self, cls: type[ExecNode]) -> cls:
        assert isinstance(self, cls)
        return self

class ExecNodeRoot(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            query: str):
        super().__init__(
            execGraph=execGraph, 
            type=ExecNodeType.ROOT,
            label=ExecNodeLabel.ROOT)

        self.query = query
        self.agent :ExecNodeAgent = None

    def setAgent(self, agent: ExecNodeAgent):
        self.agent = agent

    def getAgent(self):
        return self.agent

    def toDict(self) -> dict:
        base = super().toDict()
        base["query"] = self.query
        base["agent"] = self.agent.getId()
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            query: str,
            agent: ExecNodeAgent=None):
        execNodeRoot :ExecNodeRoot = ExecNodeRoot(
            execGraph=execGraph, 
            query=query)
        if agent:
            execNodeRoot.setAgent(agent)
        return execNodeRoot

class ExecNodeCommon(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            label: ExecNodeLabel,
            instructionId: str = None):
        super().__init__(
            execGraph=execGraph, 
            type=ExecNodeType.COMMON,
            label=label,
            instructionId=instructionId)

class ExecNodeSequence(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            label: ExecNodeLabel,
            context: dict = {}):
        super().__init__(
            execGraph=execGraph, 
            type=ExecNodeType.SEQUENCE,
            label=label)
        
        self.instrs = []
        self.context = context

    def addInstruct(self, instruct: ExecNode):
        self.instrs.append(instruct)

    def getInstructs(self):
        return self.instrs

    def toDict(self) -> dict:
        base = super().toDict()
        base["instructions"] = [instruction.getId() for instruction in self.instrs]
        base["context"] = self.context
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            context: dict = {}):
        return ExecNodeSequence(
            execGraph=execGraph, 
            label=ExecNodeLabel.BLOCK,
            context=context)

class ExecNodeCall(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            label: ExecNodeLabel):
        super().__init__(
            execGraph=execGraph, 
            type=ExecNodeType.CALL,
            label=label)
        
class ExecNodeFor(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph):
        super().__init__(
            execGraph=execGraph, 
            type=ExecNodeType.FOR)

        self.executes = []

    def addExecute(self, execute: ExecNodeSequence):
        self.executes.append(execute)

    def getExecutes(self):
        return self.executes

    def toDict(self) -> dict:
        base = super().toDict()
        base["executes"] = [execute.getId() for execute in self.executes]
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            execNodeAgent: ExecNodeAgent):
        execNodeFor :ExecNodeFor = ExecNodeFor(execGraph)
        execNodeAgent.addInstruct(execNodeFor)
        return execNodeFor

class ExecNodeAgent(ExecNodeSequence):
    def __init__(
            self, 
            execGraph: ExecGraph,
            name: str = None):
        super().__init__(
            execGraph=execGraph,
            label=ExecNodeLabel.AGENT)

        self.name = name

    def addInstruct(self, instruct: ExecNode):
        self.instrs.append(instruct)

    def toDict(self) -> dict:
        base = super().toDict()
        base["name"] = self.name
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            callee: ExecNodeRoot | ExecNodeSkill,
            name: str):
        execNodeAgent :ExecNodeAgent = ExecNodeAgent(execGraph, name=name)
        if callee.label == ExecNodeLabel.ROOT:
            callee.setAgent(execNodeAgent)
        elif callee.label == ExecNodeLabel.SKILL:
            callee.setCalled(execNodeAgent)
        else:
            raise RuntimeError(f"Invalid callee label[{callee.label}]")
        return execNodeAgent

class ExecNodeLLM(ExecNodeCommon):
    def __init__(
            self, 
            execGraph: ExecGraph,
            instructionId: str = None):
        super().__init__(
            execGraph=execGraph,
            label=ExecNodeLabel.LLM,
            instructionId=instructionId)
        
        self.curInstruct = ""
        self.content = ""
        self.skills :List[ExecNodeSkill] = []

    def toDict(self) -> dict:
        base = super().toDict()
        base["curInstruct"] = self.curInstruct
        base["content"] = self.content
        base["skills"] = [skill.getId() for skill in self.skills]
        return base

    def setCurInstruct(self, curInstruct: str):
        self.curInstruct = curInstruct

    def getCurInstruct(self):
        return self.curInstruct

    def addContent(self, content: str):
        self.content += content

    def setContent(self, content: str):
        self.content = content

    def getContent(self):
        return self.content

    def addSkill(self, skill: ExecNodeSkill):
        self.skills.append(skill)

    def getSkills(self):
        return self.skills

    @staticmethod
    def build(
            execGraph: ExecGraph,
            execNodeSequence: ExecNodeSequence,
            instructionId: str,
            curInstruct: str = None):
        execNodeLLM :ExecNodeLLM = ExecNodeLLM(execGraph, instructionId)
        if curInstruct:
            execNodeLLM.setCurInstruct(curInstruct)
        execNodeSequence.addInstruct(execNodeLLM)
        return execNodeLLM

class ExecNodeSkill(ExecNodeCall):
    def __init__(
            self, 
            execGraph: ExecGraph):
        super().__init__(
            execGraph=execGraph,
            label=ExecNodeLabel.SKILL)

        self.skillName = ""
        self.query = ""
        self.skillResult = ""
        self.called : ExecNodeAgent | ExecNodeTool = None

    def setSkillName(self, skillName: str):
        self.skillName = skillName

    def setQuery(self, query: str):
        self.query = query

    def setSkillResult(self, skillResult: str):
        self.skillResult = skillResult

    def setCalled(self, called: ExecNodeAgent | ExecNodeTool):
        self.called = called

    def getCalled(self):
        return self.called

    def toDict(self) -> dict:
        base = super().toDict()
        base["skillName"] = self.skillName
        base["query"] = self.query
        base["skillResult"] = self.skillResult
        base["called"] = self.called.getId()
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            execNodeLLM: ExecNodeLLM,
            skillName: str, 
            query: str,
            skillResult: str,
            label: ExecNodeLabel):
        execNodeSkill = ExecNodeSkill(execGraph)
        execNodeSkill.setSkillName(skillName)
        execNodeSkill.setQuery(query)
        execNodeSkill.setSkillResult(skillResult)
        if label == ExecNodeLabel.AGENT:
            ExecNodeAgent.build(
                execGraph=execGraph,
                callee=execNodeSkill,
                name=skillName)
        elif label == ExecNodeLabel.TOOL:
            ExecNodeTool.build(
                execGraph=execGraph,
                execNodeSkill=execNodeSkill,
                name=skillName)

        execNodeLLM.addSkill(execNodeSkill)
        return execNodeSkill

class ExecNodeTool(ExecNodeCommon):
    def __init__(
            self, 
            execGraph: ExecGraph,
            name: str):
        super().__init__(
            execGraph=execGraph,
            label=ExecNodeLabel.TOOL)

        self.name = name

    def toDict(self) -> dict:
        base = super().toDict()
        base["name"] = self.name
        return base

    @staticmethod
    def build(
            execGraph: ExecGraph,
            execNodeSkill: ExecNodeSkill,
            name: str):
        execNodeTool = ExecNodeTool(execGraph, name)
        execNodeSkill.setCalled(execNodeTool)
        return execNodeTool

class ExecGraph:

    def __init__(self):
        self.nodes = {}
        self.rootNode :ExecNodeRoot = None

    def start(self, query: str):
        self.rootNode = ExecNodeRoot(self, query)
        self.nodes[self.rootNode.getId()] = self.rootNode

    def addNode(self, node: ExecNode):
        self.nodes[node.getId()] = node

    def getNode(self, nodeId: str):
        return self.nodes[nodeId]

    def getRootNode(self):
        return self.rootNode
    
    def dump(self):
        """
        将执行图转换为JSON字符串，输出嵌套的树形结构。
        """
        try:
            nodesList = []
            
            # 确保所有节点都有必要的字段
            for nodeId, node in self.nodes.items():
                try:
                    nodeDict = node.toDict()
                    
                    # 确保每个节点至少有id和type
                    if not nodeDict.get("id"):
                        nodeDict["id"] = nodeId
                    
                    if not nodeDict.get("type"):
                        nodeDict["type"] = "UNKNOWN"
                        
                    # 添加到结果列表
                    nodesList.append(nodeDict)
                except Exception as e:
                    # 如果单个节点转换失败，记录错误但继续处理其他节点
                    print(f"节点 {nodeId} 转换失败: {str(e)}")
            
            # 如果没有节点，添加一个占位节点避免前端显示空白
            if len(nodesList) == 0 and self.rootNode:
                rootDict = {
                    "id": self.rootNode.getId(),
                    "type": "ROOT",
                    "label": "ROOT"
                }
                if hasattr(self.rootNode, "query"):
                    rootDict["query"] = self.rootNode.query
                nodesList.append(rootDict)
            
            result = {"nodes": nodesList}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(f"执行图转换JSON失败: {str(e)}")
            # 返回基本结构确保前端不会崩溃
            return json.dumps({"nodes": [], "error": str(e)}, ensure_ascii=False)