from enum import Enum
import uuid

class ExecNodeType(Enum):
    ROOT = 0
    AGENT = 1
    LLM = 2
    SKILL = 3
    ASSEMBLE = 4

class ExecNodeStatus(Enum):
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3

class ExecNode:

    def __init__(
            self, 
            nodeType: ExecNodeType,
            instructionId: str = None):
        self.nodeId = self.createNodeId()
        self.nodeType = nodeType
        self.nodeStatus = ExecNodeStatus.PENDING
        self.parentNode :ExecNode = None
        self.instructionId :str = instructionId

    def createChildNode(self, nodeType: ExecNodeType):
        childNode :ExecNode = None
        if nodeType == ExecNodeType.AGENT:
            childNode = ExecNodeAgent()
        elif nodeType == ExecNodeType.LLM:
            childNode = ExecNodeLLM()
        elif nodeType == ExecNodeType.SKILL:
            childNode = ExecNodeSkill()
        elif nodeType == ExecNodeType.ASSEMBLE:
            childNode = ExecNodeAssemble()
        else:
            raise ValueError(f"Invalid node type: {nodeType}")
        
        childNode.parentNode = self
        return childNode

    def createNodeId(self):
        return f"{self.nodeType.name}_{uuid.uuid4()}"

    def setInstructionId(self, instructionId: str):
        self.instructionId = instructionId

    def getInstructionId(self):
        return self.instructionId

class ExecNodeRoot:
    def __init__(
            self, 
            query: str):
        super().__init__(ExecNodeType.ROOT)

        self.query = query
        self.nodeStatus = ExecNodeStatus.SUCCESS

class ExecNodeAgent:
    def __init__(self):
        super().__init__(ExecNodeType.AGENT)

class ExecNodeLLM:
    def __init__(
            self, 
            instructionId: str):
        super().__init__(ExecNodeType.LLM, instructionId)
        
        self.content = ""

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

class ExecNodeSkill:
    def __init__(self):
        super().__init__(ExecNodeType.SKILL)

        self.skillName = ""
        self.skillArgs = {}
        self.skillResult = ""

    def setSkillName(self, skillName: str):
        self.skillName = skillName

    def setSkillArgs(self, skillArgs: dict):
        self.skillArgs = skillArgs

    def setSkillResult(self, skillResult: str):
        self.skillResult = skillResult

    @staticmethod
    def build(
            execNodeParent: ExecNode, 
            skillName: str, 
            skillArgs: dict, 
            skillResult: str):
        execNodeSkill = execNodeParent.createChildNode(ExecNodeType.SKILL)
        execNodeSkill.setSkillName(skillName)
        execNodeSkill.setSkillArgs(skillArgs)
        execNodeSkill.setSkillResult(skillResult)
        return execNodeSkill

class ExecNodeAssemble:
    
    def __init__(self):
        super().__init__(ExecNodeType.ASSEMBLE)
        self.nodes = []

    def addNode(self, node: ExecNode):
        self.nodes.append(node)
        
    def getNodes(self):
        return self.nodes
   
class ExecGraph:

    def __init__(self):
        self.nodes = {}
        self.rootNode :ExecNodeRoot = None

    def start(self, query: str):
        self.rootNode = ExecNodeRoot(query)
        self.nodes[self.rootNode.nodeId] = self.rootNode

    def addNode(self, node: ExecNode):
        self.nodes[node.nodeId] = node

    def getNode(self, nodeId: str):
        return self.nodes[nodeId]
