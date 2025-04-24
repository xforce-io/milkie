import logging
from enum import Enum
from typing import List, Optional, Tuple
import re

from llama_index_client import ChatMessage, MessageRole
from milkie.agent.exec_graph import ExecNodeLLM, ExecNodeLabel, ExecNodeSkill, ExecNodeTool
from milkie.config.constant import SymbolEndSkill, SymbolStartSkill
from milkie.context import Context, History
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.functions.toolkits.toolkit import FuncExecRecord, Toolkit
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, WARNING
from milkie.response import Response
from milkie.types.object_type import ObjectType

logger = logging.getLogger(__name__)

class InstAnalysisResult:
    class Result(Enum):
        TOOL = 1
        ANSWER = 2
        NOANS = 3

    def __init__(
            self,
            result :Result,
            funcExecRecords: List[FuncExecRecord],
            response :str):
        self.result = result
        self.funcExecRecords = funcExecRecords
        self.response = response

class SkillTag:
    def __init__(
            self, 
            start: int, 
            end: int, 
            skillName: str, 
            query: str,
            toolkit: Toolkit = None,
            toolName: str = None):
        self.start = start
        self.end = end
        self.skillName = skillName
        self.query = query
        self.toolkit = toolkit
        self.toolName = toolName
    
    def isAgent(self):
        return self.toolkit and isinstance(self.toolkit, AgentToolkit)

def callSkill(
        context: Context,
        llm :EnhancedLLM,
        funcCall: str,
        query: str,
        preContext: str,
        skillTag: SkillTag,
        **kwargs) -> str:
    execNodeLLM = kwargs["execNode"].castTo(ExecNodeLLM)
    if skillTag.isAgent():
        execSkillNode = ExecNodeSkill.build(
            execGraph=execNodeLLM.execGraph,
            execNodeLLM=execNodeLLM,
            skillName=skillTag.skillName,
            query=funcCall.strip(),
            skillResult=None,
            label=ExecNodeLabel.AGENT)
        response = skillTag.toolkit.agent.execute(
            context=context,
            query=funcCall.strip(), 
            args=context.getVarDict().getGlobalDict(),
            **{"execNode" : execSkillNode.getCalled()})
        execSkillNode.setSkillResult(response.respStr)
        return response.respStr
    else:
        history = History()
        if preContext:
            history.addUserPrompt(preContext)

        if skillTag.toolName and skillTag.toolkit.isQueryAsArg():
            tool = skillTag.toolkit.getToolAsTools(skillTag.toolName)[0]
            response = tool.func(query)
            context.genResp(response, **kwargs)
            return response

        tools = skillTag.toolkit.getTools() if skillTag.toolName is None else skillTag.toolkit.getToolAsTools(skillTag.toolName)
        kwargs = {
            "tools" : tools, 
            "execNode" : execNodeLLM,
            "history" : history,
            "toolDetect" : True
        }
        stepLLMToolcall = StepLLMStreaming()
        stepLLMToolcall.setLLM(llm)
        stepLLMToolcall.setContext(context)
        stepLLMToolcall.setQuery(funcCall.strip())
        stepLLMToolcall.setToolkit(skillTag.toolkit)
        stepLLMToolcall.setGlobalSkillset(
            context.getGlobalContext().getEnv().getGlobalSkillset()
        )
        response = stepLLMToolcall.streamAndFormat(**kwargs)
        return response.response

class StepLLMStreaming(StepLLM):
    
    def __init__(self):
        super().__init__(
            globalContext=None,
            llm=None)

        self.needToParse = False
        self.objectTypes = None
    
    def setToolkit(self, toolkit: Toolkit):
        self.toolkit = toolkit

    def setObjectTypes(self, objectTypes: List[ObjectType]):
        self.objectTypes = objectTypes

    def setGlobalSkillset(self, globalSkillset: Skillset):
        self.globalSkillset = globalSkillset

    def setContext(self, context: Context):
        self.globalContext = context.globalContext
        self.context = context

    def setQuery(self, query: str):
        self.query = query

    def makePrompt(self, useTool: bool = False, **args) -> str:
        return self.query
    
    def formatResult(self, result: Response, **kwargs) -> str:
        allDict = self.context.getVarDict().getAllDict()

        streamingProcessor = StreamingProcessor(
            globalSkillset=self.globalSkillset,
            context=self.context,
            llm=self.llm,
            messages=self._messages)
        toolUsed, result = streamingProcessor.readFromGen(
            response=result,
            **kwargs)
        if toolUsed and not self.objectTypes:
            funcExecRecords = self.toolkit.exec(
                [(toolUsed, result)], 
                allDict,
                needToParse=self.needToParse)
            self.context.genResp(funcExecRecords[0].result, **kwargs)

            execNodeLLM = kwargs["execNode"]
            assert execNodeLLM.label == ExecNodeLabel.LLM
            
            execNode = ExecNodeSkill.build(
                execGraph=execNodeLLM.execGraph,
                execNodeLLM=execNodeLLM,
                skillName=toolUsed,
                query=self.query,
                skillResult=funcExecRecords[0].result,
                label=ExecNodeLabel.TOOL)
            
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=funcExecRecords[0].result)

        self.context.genResp(result, end="", **kwargs)
        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result)

class QueryCntToCallSkill:
    def __init__(self) -> None:
        self.cnts = {}

    def addQuery(self, query :str):
        cnt = self.cnts.get(query)
        if cnt == None:
            self.cnts[query] = 1
        else :
            self.cnts[query] += 1

    def checkValid(self, query :str) -> bool:
        cnt = self.cnts.get(query)
        return cnt is None or cnt < 3

class StreamingProcessor:
    def __init__(self,
                 globalSkillset: Skillset,
                 context: Context,
                 llm: EnhancedLLM,
                 messages: list[ChatMessage]):
        self.globalSkillset = globalSkillset
        self.context = context
        self.accuResults = []
        self.allResults = []
        self.currentSentence = []
        self.currentTools = []
        self.response = None
        self.llm = llm
        self.messages = messages
        self.streamReset = False
        self.queryCntToCallSkill = QueryCntToCallSkill()

    def readFromGen(
            self,
            response: Response, 
            **kwargs) -> Tuple[Optional[str], str]:
        self.response = response
        skillResp = None

        try:
            if not self.response.respGen:
                WARNING("No response generator found")
                return None, ""

            while True:
                for chunk in self.response.respGen:
                    content = None
                    if hasattr(chunk.raw, "tool_calls") and \
                            isinstance(chunk.raw.tool_calls, list) and \
                            len(chunk.raw.tool_calls) > 0:
                        toolCalls = chunk.raw.tool_calls
                        if len(self.currentTools) == 0:
                            self.currentTools = [{
                                "name": "",
                                "args": ""
                            }] * len(toolCalls)
                        
                        for i, toolCall in enumerate(toolCalls):
                            if toolCall.function.name:
                                self.currentTools[i]["name"] = toolCall.function.name
                            if toolCall.function.arguments:
                                self.currentTools[i]["args"] += toolCall.function.arguments
                    elif hasattr(chunk.raw, "delta") and hasattr(chunk.raw.delta, "content"):
                        content = chunk.raw.delta.content
                    elif hasattr(chunk.raw, "delta") and hasattr(chunk.raw.delta, "reasoning_content"):
                        content = chunk.raw.delta.reasoning_content
                    elif hasattr(chunk.raw, "content") and chunk.raw.content:
                        content = chunk.raw.content
                    elif hasattr(chunk.raw, "reasoning_content") and chunk.raw.reasoning_content:
                        content = chunk.raw.reasoning_content

                    if content and "toolDetect" not in kwargs:
                        skillResp = self.processDeltaContent(
                            deltaContent=content,
                            **kwargs)
                        if skillResp:
                            break

                if self.streamReset:
                    self.streamReset = False
                    continue

                if self.currentSentence:
                    self._clearCurSentence(**kwargs)
                break

        except Exception as e:
            logger.error(f"Error in readFromGen: {e}", exc_info=True)
            raise

        if len(self.currentTools) > 0:
            return self.currentTools[0]["name"], self.currentTools[0]["args"]
        else:
            self.allResults.extend(self.accuResults)
            return None, "".join(self.allResults)

    def processDeltaContent(
            self,
            deltaContent: str,
            **kwargs) -> str:
        if not self.globalSkillset:
            self._addCurSentence(deltaContent, **kwargs)
            return None
            
        curSentenceWithDelta = self._getCurSentence() + deltaContent
        skillTag = self._findSkillTag(curSentenceWithDelta, 0)
        if not skillTag or not self.queryCntToCallSkill.checkValid(skillTag.query):
            self._addCurSentence(deltaContent, **kwargs)
            return None

        self.queryCntToCallSkill.addQuery(skillTag.query)

        self._addCurSentence(deltaContent, **kwargs)
        
        self._clearCurSentence(**kwargs)
        return self._callSkillAndResetStream(
            sentence=skillTag.query, 
            preContext=curSentenceWithDelta[:skillTag.start], 
            skillTag=skillTag,
            **kwargs)

    def _getCurSentence(self) -> str:
        return "".join(self.currentSentence).strip()

    def _getCurAccuResults(self) -> str:
        return "".join(self.accuResults).strip()

    def _addCurSentence(self, content: str, **kwargs):
        self.currentSentence.append(content)
        self.context.genResp(content, end="", **kwargs)
        execNodeLLM = kwargs["execNode"].castTo(ExecNodeLLM)
        execNodeLLM.addContent(content)

    def _clearCurSentence(self, **kwargs):
        sentence = self._getCurSentence()
        self.accuResults.append(sentence)
        self.currentSentence = []

    def _callSkillAndResetStream(
            self, 
            sentence: str, 
            preContext: str, 
            skillTag: SkillTag,
            **kwargs) -> str:
        skillLlm = self.context.getGlobalContext().settings.getLLMSkill()
        skillResp = callSkill(
            context=self.context,
            llm=skillLlm,
            funcCall=sentence,
            query=sentence,
            preContext=preContext,
            skillTag=skillTag,
            **kwargs)

        answer = f"<answer>{skillResp}</answer>"
        resultWithEndMark = f"{self._getCurAccuResults()}{answer}"
        if not self.llm.prefix_complete:
            newMessage = resultWithEndMark + "我们继续"
            if len(self.messages) > 0 and self.messages[-1].role == MessageRole.ASSISTANT:
                self.messages[-1] = ChatMessage(
                    role=MessageRole.ASSISTANT, 
                    content=self.messages[-1].content + newMessage)
            else:
                self.messages += [
                    ChatMessage(                                                        
                        role=MessageRole.ASSISTANT, 
                        content=newMessage),   
                ]
        elif len(self.messages) > 1 and self.messages[-1].role == MessageRole.ASSISTANT:
            self.messages[-1] = ChatMessage(
                role=MessageRole.ASSISTANT, 
                content=self.messages[-1].content + resultWithEndMark, 
                additional_kwargs={"prefix" : True})
        else:
            if len(self.messages) > 0 and self.messages[-1].role == MessageRole.ASSISTANT:
                self.messages[-1] = ChatMessage(
                    role=MessageRole.ASSISTANT, 
                    content=self.messages[-1].content + resultWithEndMark,
                    additional_kwargs={"prefix" : True})
            else:
                self.messages += [
                    ChatMessage(                                                        
                        role=MessageRole.ASSISTANT, 
                        content=resultWithEndMark,
                        additional_kwargs={"prefix" : True}),   
                ]
        self.allResults.extend(self.accuResults)
        self.allResults.extend([answer])
        self.accuResults = []

        self.response.respGen = self.llm.stream(self.messages)
        self.streamReset = True
        return answer

    def _findSkillTag(
            self, 
            content: str, 
            startIndex: int) -> Optional[SkillTag]:
        pattern = r'<([\w\d._]+)>([\s\S]*)</[\w\d._]+>'
        match = re.search(pattern, content[startIndex:])
        if not match:
            return None

        skillName = match.group(1)
        query = match.group(2)
        globalSkillset = self.context.getGlobalContext().getEnv().getGlobalSkillset()
        for toolkit in globalSkillset.skillset:
            if isinstance(toolkit, AgentToolkit):
                if skillName == toolkit.getName():
                    return SkillTag(
                        start=startIndex + match.start(),
                        end=startIndex + match.end(),
                        skillName=skillName,
                        query=query,
                        toolkit=toolkit)
            else:
                for toolName, _ in toolkit.getToolDescs().items():
                    if skillName == toolName or \
                            skillName == f"{toolkit.getName()}":
                        return SkillTag(
                            start=startIndex + match.start(),
                            end=startIndex + match.end(),
                            skillName=skillName,
                            query=query,
                            toolkit=toolkit)
                    elif skillName == f"{toolkit.getName()}.{toolName}":
                        return SkillTag(
                            start=startIndex + match.start(),
                            end=startIndex + match.end(),
                            skillName=toolkit.getName(),
                            toolName=toolName,
                            query=query,
                            toolkit=toolkit)
        return None
    