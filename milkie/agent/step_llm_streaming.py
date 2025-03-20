import logging
from enum import Enum
from typing import List, Optional, Tuple

from llama_index_client import ChatMessage, MessageRole
from milkie.agent.exec_graph import ExecNodeLLM, ExecNodeSkill, ExecNodeTool
from milkie.config.constant import SymbolEndSkill
from milkie.context import Context, History
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.functions.toolkits.toolkit import FuncExecRecord, Toolkit
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, WARNING
from milkie.response import Response

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

def useSkill(
        context: Context,
        llm :EnhancedLLM,
        funcCall: str,
        preContext: str,
        **kwargs) -> Tuple[str, str]:
    stepLLMToolcall = StepLLMStreaming()
    globalSkillset = context.getGlobalContext().getEnv().getGlobalSkillset()
    for toolkit in globalSkillset.skillset:
        if isinstance(toolkit, AgentToolkit):
            if not funcCall.startswith(toolkit.getName()) :
                continue

            execNode = ExecNodeSkill.build(
                execNodeParent=kwargs["execNode"], 
                skillName=toolkit.getName(),
                query=funcCall[len(toolkit.getName()):].strip(),
                skillArgs=context.getVarDict().getGlobalDict())
            response = toolkit.agent.execute(
                context=context,
                query=funcCall[len(toolkit.agent.name):].strip(), 
                args=context.getVarDict().getGlobalDict(),
                **{"execNode" : execNode})
            execNode.setSkillResult(response.respStr)
            return toolkit.getName(), response.respStr
        else:
            for toolName, _ in toolkit.getToolDescs().items():
                if not funcCall.startswith(toolName) and \
                        not funcCall.startswith(f"{toolkit.getName()}.{toolName}") and \
                        not funcCall.startswith(toolkit.getName()):
                    continue

                history = History()
                history.addHistoryUserPrompt(preContext)

                execNode = ExecNodeSkill.build(
                    execNodeParent=kwargs["execNode"], 
                    skillName=f"{toolkit.getName()}.{toolName}",
                    query=funcCall[len(toolkit.getName()):].strip(),
                    skillArgs=context.getVarDict().getGlobalDict())

                kwargs = {
                    "tools" : toolkit.getTools(), 
                    "execNode" : execNode,
                    "history" : history
                }
                stepLLMToolcall.setLLM(llm)
                stepLLMToolcall.setContext(context)
                stepLLMToolcall.setQuery(funcCall[len(toolkit.getName()):].strip())
                stepLLMToolcall.setToolkit(toolkit)
                stepLLMToolcall.setGlobalSkillset(globalSkillset)
                response = stepLLMToolcall.streamAndFormat(**kwargs)

                execNode.setSkillResult(response.response)
                return toolkit.getName(), response.response
    return None, ""

class StepLLMStreaming(StepLLM):
    
    def __init__(self):
        super().__init__(
            globalContext=None,
            llm=None)

        self.needToParse = False
    
    def setToolkit(self, toolkit: Toolkit):
        self.toolkit = toolkit

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
        if toolUsed:
            funcExecRecords = self.toolkit.exec(
                [(toolUsed, result)], 
                allDict,
                needToParse=self.needToParse)
            self.context.genResp(funcExecRecords[0].result, **kwargs)

            ExecNodeTool.build(
                execNodeParent=kwargs["execNode"], 
                toolName=toolUsed, 
                query=self.query,
                toolArgs=result, 
                toolResult=funcExecRecords[0].result)
            
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=funcExecRecords[0].result)

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result)

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
        self.sentences_to_detect_skill = []

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

                    if content:
                        skillResp = self.processDeltaContent(
                            deltaContent=content,
                            **kwargs)
                        if skillResp:
                            break

                if self.streamReset:
                    self.streamReset = False
                    continue

                if self.currentSentence:
                    skillResp = self.processSentence(**kwargs)
                    if not skillResp:
                        break

                    self.accuResults.append(skillResp)
                else:
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
        endIndex = -len(SymbolEndSkill)
        while True:
            startIndex = endIndex + len(SymbolEndSkill)
            i = startIndex
            endIndex = -1
            while i < len(deltaContent):
                if deltaContent[i:].startswith(SymbolEndSkill):
                    endIndex = i
                    break
                i += 1

            skillResp = ""
            if endIndex >= 0:
                currentPart = deltaContent[startIndex:endIndex+len(SymbolEndSkill)]
                self.currentSentence.append(currentPart)
                self.context.genResp(currentPart, end="", **kwargs)
                if kwargs["execNode"] and isinstance(kwargs["execNode"], ExecNodeLLM):
                    kwargs["execNode"].addContent(currentPart)
                
                skillResp = self.processSentence(**kwargs)
                if skillResp:
                    return skillResp
            else:
                self.currentSentence.append(deltaContent[startIndex:])
                self.context.genResp(deltaContent[startIndex:], end="", **kwargs)
                if kwargs["execNode"] and isinstance(kwargs["execNode"], ExecNodeLLM):
                    kwargs["execNode"].addContent(deltaContent[startIndex:])
                return None

    def processSentence(self, **kwargs) -> str:
        sentence = "".join(self.currentSentence).strip()
        self.accuResults.append(sentence)
        self.currentSentence = []
        if not self.globalSkillset:
            return None
        
        startIndex = 0
        while True:
            idx = sentence.find("@", startIndex)
            if idx == -1:
                break

            resp = self._processSentenceStartWithAt(
                sentence=sentence[idx:], 
                preContext=sentence[:idx], 
                **kwargs)
            if resp:
                return resp

            startIndex = idx + 1
        return None

    def _processSentenceStartWithAt(self, sentence: str, preContext: str, **kwargs) -> str:
        if sentence in self.sentences_to_detect_skill:
            INFO(logger, f"sentence[{sentence}] in sentences_to_detect_skill[{self.sentences_to_detect_skill}]")
            return None

        skillLlm = self.llm if not self.llm.reasoner_model else self.context.getGlobalContext().settings.llmDefault
        skillName, skillResp = useSkill(
            context=self.context,
            llm=skillLlm,
            funcCall=sentence[1:],
            preContext=preContext,
            **kwargs)
        if not skillName:
            return None

        self.sentences_to_detect_skill.append(sentence)

        endMark = f" @{skillName} END\n"
        self.context.genResp(f"<<<{endMark.strip()}>>>\n", end="", **kwargs)

        if not self.llm.prefix_complete:
            self.messages += [
                ChatMessage(                                                        
                    role=MessageRole.ASSISTANT, 
                    content="".join(self.accuResults) + skillResp + endMark + "我们继续"),   
            ]
        elif len(self.messages) > 1 and self.messages[-1].role == MessageRole.ASSISTANT:
            self.messages[-1] = ChatMessage(
                role=MessageRole.ASSISTANT, 
                content=self.messages[-1].content + "".join(self.accuResults) + skillResp + endMark, 
                additional_kwargs={"prefix" : True})
        else:
            self.messages += [
                ChatMessage(                                                        
                    role=MessageRole.ASSISTANT, 
                    content="".join(self.accuResults) + skillResp + endMark, 
                    additional_kwargs={"prefix" : True}),   
            ]
        self.allResults.extend(self.accuResults)
        self.allResults.extend([skillResp, endMark])
        self.accuResults = []

        self.response.respGen = self.llm.stream(self.messages)
        self.streamReset = True
        return skillResp + endMark