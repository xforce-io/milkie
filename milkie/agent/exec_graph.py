from __future__ import annotations
from enum import Enum
import uuid
import json

from milkie.log import ERROR

class ExecNodeType(Enum):
    ROOT = 0
    AGENT = 1
    LLM = 2
    SKILL = 3
    ASSEMBLE = 4
    TOOL = 5

class ExecNode:

    def __init__(
            self, 
            execGraph: ExecGraph,
            nodeType: ExecNodeType,
            instructionId: str = None):
        self.execGraph = execGraph
        self.nodeType = nodeType
        self.nodeId = self.createNodeId()
        self.parentNode :ExecNode = None
        self.instructionId :str = instructionId

        self.execGraph.addNode(self)

    def toDict(self) -> dict:
        """将节点转换为字典表示"""
        base = {
            "id": self.nodeId,
            "type": self.nodeType.name,
        }
        if self.instructionId:
            base["instructionId"] = self.instructionId
        return base

    def createChildNode(self, nodeType: ExecNodeType):
        childNode :ExecNode = None
        if nodeType == ExecNodeType.AGENT:
            childNode = ExecNodeAgent(self.execGraph)
        elif nodeType == ExecNodeType.LLM:
            childNode = ExecNodeLLM(self.execGraph)
        elif nodeType == ExecNodeType.SKILL:
            childNode = ExecNodeSkill(self.execGraph)
        elif nodeType == ExecNodeType.ASSEMBLE:
            childNode = ExecNodeAssemble(self.execGraph)
        elif nodeType == ExecNodeType.TOOL:
            childNode = ExecNodeTool(self.execGraph)
        else:
            raise ValueError(f"Invalid node type: {nodeType}")
        
        childNode.parentNode = self
        return childNode

    def createNodeId(self):
        return f"{self.nodeType.name}_{uuid.uuid4()}"

    def setInstructionId(self, instructionId: str):
        self.instructionId = instructionId

    def getNodeId(self):
        return self.nodeId

    def getInstructionId(self):
        return self.instructionId

class ExecNodeRoot(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            query: str):
        super().__init__(execGraph, ExecNodeType.ROOT)

        self.query = query

    def toDict(self) -> dict:
        base = super().toDict()
        base["query"] = self.query
        return base

class ExecNodeAgent(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            name: str = None):
        super().__init__(execGraph, ExecNodeType.AGENT)

        self.name = name

    def toDict(self) -> dict:
        base = super().toDict()
        base["name"] = self.name
        return base

    @staticmethod
    def build(
            execNodeParent: ExecNode, 
            name: str):
        execNodeAgent :ExecNodeAgent = execNodeParent.createChildNode(ExecNodeType.AGENT)
        execNodeAgent.name = name
        return execNodeAgent

class ExecNodeLLM(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph,
            instructionId: str = None):
        super().__init__(execGraph, ExecNodeType.LLM, instructionId)
        
        self.content = ""

    def toDict(self) -> dict:
        base = super().toDict()
        base["content"] = self.content
        return base

    def addContent(self, content: str):
        self.content += content

    def setContent(self, content: str):
        self.content = content

    def getContent(self):
        return self.content

    @staticmethod
    def build(
            execNodeParent: ExecNode, 
            instructionId: str):
        execNodeLLM :ExecNodeLLM = execNodeParent.createChildNode(ExecNodeType.LLM)
        execNodeLLM.setInstructionId(instructionId)
        return execNodeLLM

class ExecNodeSkill(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph):
        super().__init__(execGraph, ExecNodeType.SKILL)

        self.skillName = ""
        self.query = ""
        self.skillArgs = {}
        self.skillResult = ""

    def toDict(self) -> dict:
        base = super().toDict()
        base["skillName"] = self.skillName
        base["query"] = self.query
        base["skillResult"] = self.skillResult
        base["skillArgs"] = self.skillArgs
        return base

    def setSkillName(self, skillName: str):
        self.skillName = skillName

    def setQuery(self, query: str):
        self.query = query

    def setSkillArgs(self, skillArgs: dict):
        self.skillArgs = skillArgs

    def setSkillResult(self, skillResult: str):
        self.skillResult = skillResult

    @staticmethod
    def build(
            execNodeParent: ExecNode, 
            skillName: str, 
            query: str,
            skillArgs: dict, 
            skillResult: str = None):
        execNodeSkill = execNodeParent.createChildNode(ExecNodeType.SKILL)
        execNodeSkill.setSkillName(skillName)
        execNodeSkill.setQuery(query)
        execNodeSkill.setSkillArgs(skillArgs)
        execNodeSkill.setSkillResult(skillResult)
        return execNodeSkill

class ExecNodeTool(ExecNode):
    def __init__(
            self, 
            execGraph: ExecGraph):
        super().__init__(execGraph, ExecNodeType.TOOL)

        self.toolName = ""
        self.query = ""
        self.toolArgs = {}
        self.toolResult = ""

    def toDict(self) -> dict:
        base = super().toDict()
        base["toolName"] = self.toolName
        base["query"] = self.query
        base["toolArgs"] = self.toolArgs
        base["toolResult"] = self.toolResult
        return base

    def setToolName(self, toolName: str):
        self.toolName = toolName

    def setQuery(self, query: str):
        self.query = query

    def setToolArgs(self, toolArgs: dict):
        self.toolArgs = toolArgs

    def setToolResult(self, toolResult: str):
        self.toolResult = toolResult

    @staticmethod
    def build(
            execNodeParent: ExecNode, 
            toolName: str, 
            query: str, 
            toolArgs: dict,
            toolResult: str = None):
        execNodeTool = execNodeParent.createChildNode(ExecNodeType.TOOL)
        execNodeTool.setToolName(toolName)
        execNodeTool.setQuery(query)
        execNodeTool.setToolArgs(toolArgs)
        execNodeTool.setToolResult(toolResult)
        return execNodeTool

class ExecNodeAssemble(ExecNode):
    
    def __init__(
            self, 
            execGraph: ExecGraph):
        super().__init__(execGraph, ExecNodeType.ASSEMBLE)
        self.nodes = []

    def toDict(self) -> dict:
        base = super().toDict()
        base["assembleNodes"] = [node.toDict() for node in self.nodes]
        return base

    def addNode(self, node: ExecNode):
        self.nodes.append(node)
        
    def getNodes(self):
        return self.nodes
   
class ExecGraph:

    def __init__(self):
        self.nodes = {}
        self.rootNode :ExecNodeRoot = None

    def start(self, query: str):
        self.rootNode = ExecNodeRoot(self, query)
        self.nodes[self.rootNode.getNodeId()] = self.rootNode

    def addNode(self, node: ExecNode):
        self.nodes[node.getNodeId()] = node

    def getNode(self, nodeId: str):
        return self.nodes[nodeId]

    def getRootNode(self):
        return self.rootNode
    
    def dump(self):
        """
        将执行图转换为JSON字符串，输出嵌套的树形结构。
        """
        def build_tree(node_id, visited=None):
            """递归构建树结构"""
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return None
                
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if not node:
                return None
            
            result = node.toDict()
            children = []
            
            # 处理普通的父子关系
            for child_id, child in self.nodes.items():
                if hasattr(child, 'parentNode') and child.parentNode and child.parentNode.nodeId == node_id:
                    child_tree = build_tree(child_id, visited)
                    if child_tree:
                        children.append(child_tree)
            
            if children:
                result["children"] = children
            
            return result

        tree = build_tree(self.rootNode.nodeId)
        return json.dumps(tree, ensure_ascii=False, indent=2)