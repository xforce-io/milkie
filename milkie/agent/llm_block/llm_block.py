from __future__ import annotations
from abc import abstractmethod
from enum import Enum

import json
import logging
import re
import time
from typing import Any, Callable, List, Optional, Tuple
import uuid

from llama_index_client import ChatMessage, MessageRole

from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNode, ExecNodeLLM, ExecNodeSkill
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.config.constant import *
from milkie.context import Context
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.functions.toolkits.toolkit import Toolkit, FuncExecRecord
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.prompt.prompt import Loader
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, DEBUG
from milkie.response import Response
from milkie.trace import stdout
from milkie.utils.commons import addDict
from milkie.utils.data_utils import codeToLines, restoreVariablesInDict, restoreVariablesInStr

logger = logging.getLogger(__name__)

class AnswerResult:
    class Result(Enum):
        ANSWER = 1
        NOANS = 2 
    
    def __init__(
            self,
            result :Result,
            response :str) -> None:
        self.result = result
        self.response = response 

class ThinkingResult:
    class Result(Enum):
        THINK = 1
        ERROR = 2

    def __init__(
            self,
            result :Result,
            response :str) -> None:
        self.result = result
        self.response = response

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

    @abstractmethod
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
            stdout(funcExecRecords[0].result, **kwargs)

            ExecNodeSkill.build(
                execNodeParent=kwargs["execNode"], 
                skillName=toolUsed, 
                skillArgs=allDict, 
                skillResult=funcExecRecords[0].result)
            
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=f"{toolUsed}({result})")

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result)

class StepLLMInstrAnalysis(StepLLMStreaming):
    def __init__(
            self, 
            llmBlock :LLMBlock, 
            instruction: Instruction) -> None:
        super().__init__(
            llmBlock=llmBlock,
            llm=llmBlock.context.globalContext.settings.llmDefault)

        self.instruction = instruction
        self.needToParse = instruction.formattedInstruct.find("{") >= 0

        self.setToolkit(instruction.toolkit)
        self.setGlobalSkillset(llmBlock.globalSkillset)
        self.setContext(llmBlock.context)

    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        systemPrompt = super().makeSystemPrompt(args=args, **kwargs)
        if "skills" in kwargs:
            systemPrompt += f"""
            注意：拥有技能如下(skillname -> skilldesc)
            ```
            """
            for name, skill in kwargs["skills"].items():
                if isinstance(skill, AgentToolkit):
                    systemPrompt += f"{name} -> {skill.getDesc()}\n"
                else:
                    for toolName, toolDesc in skill.getToolDescs().items():
                        systemPrompt += f"{name}.{toolName} -> \n{toolDesc}\n"
            systemPrompt += '''```\n
            如果需要使用技能，请使用 "@skillname ((技能参数))" 来调用技能。
            '''
        return systemPrompt
        
    def makePrompt(self, useTool: bool = False, **args) -> str:
        result = self.instruction.formattedInstruct
        if self.instruction.syntaxParser.getNormalFormat():
            result += f"""
            请按照下述语义严格以 jsonify 格式输出结果：{self.instruction.syntaxParser.getJsonOutputSyntax()}，现在请直接输出 json:
            """
        elif not useTool:
            result += f"""
            请直接输出结果，不要输出任何其他内容:
            """
        return result

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
        self.stepLLMToolcall = StepLLMStreaming()

    def readFromGen(
            self,
            response: Response, 
            **kwargs) -> Tuple[Optional[str], str]:
        self.response = response
        skillResp = None

        try:
            if not self.response.respGen:
                logger.warning("No response generator found")
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

                stdout("", info=True, flush=True, **kwargs)

                if self.currentSentence:
                    skillResp = self.processSentence()
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
                stdout(currentPart, info=True, end="", flush=True, **kwargs)
                kwargs["execNode"].addContent(currentPart)
                
                skillResp = self.processSentence()
                if skillResp:
                    return skillResp
            else:
                self.currentSentence.append(deltaContent[startIndex:])
                stdout(deltaContent[startIndex:], info=True, end="", flush=True, **kwargs)
                kwargs["execNode"].addContent(deltaContent[startIndex:])
                return None

    def processSentence(self) -> str:
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

            resp = self._processSentenceStartWithAt(sentence[idx:])
            if resp:
                return resp

            startIndex = idx + 1
        return None

    def _processSentenceStartWithAt(self, sentence: str) -> str:
        if sentence in self.sentences_to_detect_skill:
            INFO(logger, f"sentence[{sentence}] in sentences_to_detect_skill[{self.sentences_to_detect_skill}]")
            return None
        
        self.stepLLMToolcall.setGlobalSkillset(self.globalSkillset)
        self.stepLLMToolcall.setContext(self.context)
        self.stepLLMToolcall.setQuery(sentence)
        self.stepLLMToolcall.setLLM(self.llm if not self.llm.reasoner_model else self.context.getGlobalContext().settings.llmDefault)
        
        skillName, skillResp = self._useSkill(sentence)
        if not skillName:
            return None

        self.sentences_to_detect_skill.append(sentence)

        endMark = f" @{skillName} END\n"
        stdout(f"<<<{endMark.strip()}>>>\n", info=True, end="", flush=True)

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

    def _useSkill(self, query: str) -> Tuple[str, str]:
        for toolkit in self.globalSkillset.skillset:
            if isinstance(toolkit, AgentToolkit):
                if not query[1:].startswith(toolkit.getName()) :
                    continue

                response = toolkit.agent.execute(
                    query=query[len(toolkit.agent.name)+1:].strip(), 
                    args=self.context.getVarDict().getGlobalDict())
                return toolkit.getName(), response.respStr
            else:
                for toolName, _ in toolkit.getToolDescs().items():
                    if not query[1:].startswith(toolName) and \
                            not query[1:].startswith(f"{toolkit.getName()}.{toolName}"):
                        continue

                    kwargs = {"tools" : toolkit.getTools()}
                    self.stepLLMToolcall.setToolkit(toolkit)
                    response = self.stepLLMToolcall.streamAndFormat(**kwargs)
                    return toolkit.getName(), response.response
            
        return None, ""

class InstructResult:
    def __init__(
            self, 
            response :Response,
            logMessage :str = None,
            goto :str = None,
            useTool :bool = False):
        self.response = response
        self.goto = goto
        self.useTool = useTool
        self.logMessage = logMessage

    def isRet(self):
        return self.response.respStr == KeyRet

    def isNext(self):
        return self.response.respStr == KeyNext

    def getLogMessage(self):
        return self.logMessage

from milkie.agent.llm_block.syntax_parser import SyntaxParser

class Instruction:
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            curInstruct: str,
            label: str = None,
            observation: str = None,
            prev = None) -> None:
        self.id = self._createId()
        self.llmBlock = llmBlock
        self.syntaxParser = SyntaxParser(
            label=label,
            settings=llmBlock.context.globalContext.settings,
            instruction=curInstruct, 
            repoFuncs=llmBlock.repoFuncs)
        self.curInstruct = self.syntaxParser.getInstruction()
        self.formattedInstruct = self.curInstruct
        self.onlyFuncCall = False
        self.isNaiveType = False
        self.noCache = False
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.varDict = llmBlock.getVarDict()

        self.stepInstAnalysis = StepLLMInstrAnalysis(
            llmBlock=llmBlock,
            instruction=self)
        self.llm :EnhancedLLM = None
        self.reasoning :Reasoning = None
        self.toolkit :Toolkit = None

    def getId(self):
        return self.id

    def execute(
            self, 
            execNode: ExecNode,
            args :dict, 
            **kwargs) -> InstructResult:
        try:
            self._formatCurInstruct(args)
        except Exception as e:
            raise RuntimeError(f"fail parse instruct[{self.curInstruct}]: {str(e)}")

        if self.noCache:
            kwargs["no_cache"] = True

        if self.onlyFuncCall or self.isNaiveType:
            # in this case, the response is the result of the only function call, or a naive type
            return self._processNaiveType(args, **kwargs)

        useTool = (self._getToolkit() != None)
        logTypes = []
        if self.syntaxParser.flag == SyntaxParser.Flag.CODE:
            return self._processGenCode(logType="code", args=args, **kwargs)
        elif self.syntaxParser.flag == SyntaxParser.Flag.IF:
            return self._processGenCode(logType="if", args=args, **kwargs)
        elif self.syntaxParser.flag == SyntaxParser.Flag.PY:
            return self._processPyCode(args=args, **kwargs)
        elif self.syntaxParser.flag == SyntaxParser.Flag.CALL:
            return self._processCall(args=args, **kwargs)
        elif self.syntaxParser.flag == SyntaxParser.Flag.RET and self.syntaxParser.returnVal:
            return self._processRet(args=args, **kwargs)

        args["prompt_flag"] = None

        if useTool:
            kwargs["tools"] = self._getToolkit().getTools()
        
        kwargs["globalSkillset"] = self.llmBlock.globalSkillset
        
        self.stepInstAnalysis.setLLM(self._getCurLLM())
        instAnalysisResult = self.stepInstAnalysis.streamAndFormat(
            reasoning=self.reasoning,
            args=args,
            **kwargs)

        if instAnalysisResult.result == InstAnalysisResult.Result.ANSWER:
            logTypes.append("ans")
            return self._createResult(
                response=instAnalysisResult.response, 
                useTool=False,
                goto=None,
                analysis=instAnalysisResult.response,
                logType="/".join(logTypes))

        return self._createResult(
            response=instAnalysisResult.funcExecRecords[0].result,
            useTool=True,
            goto=None,
            analysis=instAnalysisResult.response,
            answer=instAnalysisResult.funcExecRecords[0].result,
            logType="tool",
            toolName=instAnalysisResult.funcExecRecords[0].tool.get_function_name())
        
    def _createId(self):
        return f"{self.llmBlock.agentName}_{self.label}_{uuid.uuid4()}"
        
    def __str__(self) -> str:
        return f"Instruction({self.label}): {self.formattedInstruct}"
        
    def _processNaiveType(self, args: dict, **kwargs):
        stdout(self.formattedInstruct, args=args, **kwargs)
        return self._createResult(
            response=self.formattedInstruct,
            useTool=False,
            goto=None,
            analysis=self.curInstruct,
            logType="naive")

    def _processGenCode(self, logType: str, args: dict, **kwargs):

        def genCodeAndRun(instruction: Instruction, theArgs: dict):
            result = instruction.llmBlock.toolkit.genCodeAndRun(
                instruction=instruction.formattedInstruct,
                varDict=instruction.varDict.getAllDict(),
                no_cache=theArgs["no_cache"],
                **kwargs)
            instruction.varDict.setLocal(KeywordCurrent, result)
            return Response.buildFrom(result if result else "")

        return self._processWithRetry(genCodeAndRun, args=args, logType=logType)

    def _processPyCode(self, args: dict, **kwargs):

        def preprocessPyInstruct(instruct: str):
            instruct = instruct.lstrip() \
                .replace("$varDict", "self.varDict") \
                .replace(KeyNext, f'"{KeyNext}"') \
                .replace(KeyRet, f'"{KeyRet}"')
            return instruct

        result = self.llmBlock.toolkit.runCode(
            preprocessPyInstruct(self.formattedInstruct),
            self.varDict.getAllDict())
        if result == None or not Response.isNaivePyType(result):
            result = ""

        stdout(result, args=args, **kwargs)
        return self._createResult(
            result,
            useTool=False,
            goto=None,
            analysis=self.curInstruct,
            logType="py")
    
    def _processCall(self, args: dict, **kwargs):
        def callFunc(instruction: Instruction, theArgs: dict):
            args = {**instruction.llmBlock.context.getVarDict().getAllDict(), **theArgs["args"]}
            return instruction.llmBlock.context.getEnv().execute(
                agentName=instruction.syntaxParser.callObj,
                query=theArgs["query"].format(**args),
                args=args)
        return self._processWithRetry(lambdaFunc=callFunc, args=args, logType="call")

    def _processRet(self, args: dict, **kwargs):
        stdout(self.formattedInstruct, args=args, **kwargs)
        return self._createResult(
            response=self.formattedInstruct,
            useTool=False,
            goto=None,
            analysis=self.curInstruct,
            logType="ret")

    def _processWithRetry(
            self, 
            lambdaFunc: Callable[[Instruction, dict], Any],
            args: dict, 
            logType: str,
            **kwargs):
        query = self.syntaxParser.callArg
        MaxRetry = 3
        for i in range(MaxRetry):
            stdout(f"[trial {i}] execute call {self.syntaxParser.callObj} with query {query} ==> ", args=args, **kwargs)
                
            resp = lambdaFunc(self, {
                "query": query,
                "args": args,
                "no_cache": i != 0
            })

            instrOutput = self.syntaxParser.getInstrOutput()
            instrOutput.processOutputAndStore(
                output=resp.resp,
                stepLLMExtractor=self.llmBlock.stepLLMExtractor,
                varDict=self.varDict,
                toolkit=self._getToolkit(),
                retry=True,
                contextLen=self.syntaxParser.model.getContextWindow())
            if not instrOutput.hasError():
                stdout(f"==> {resp.resp}", args=args, **kwargs)
                return self._createResult(
                    response=resp,
                    useTool=False,
                    goto=None,
                    analysis=self.curInstruct,
                    logType=logType)
            
            query = "\n".join(instrOutput.getErrMsgs() + [query] if query else [])
        raise RuntimeError(f"fail execute instruction[{self.curInstruct}] error[{instrOutput.getErrMsgs()}]")

    def _createResult(
            self, 
            response: Any, 
            useTool :bool, 
            goto :str,
            logType: str,
            **logData) -> InstructResult:
        logMessage = f"instrExec({self.label}|{self._getCurLLM().model_name}|{logType}): instr[{self.formattedInstruct}] "
        if logType == "tool":
            logMessage += f"tool[{logData.get('toolName', '')}] "
        logMessage += f"ans[{response}]"
        
        DEBUG(logger, f"var_dict[{self.varDict}]")
        
        self.llmBlock.context.reqTrace.add("instruction", logData)
        if type(response) == str:
            #normal response
            if self.syntaxParser.flag == SyntaxParser.Flag.RET and self.syntaxParser.returnVal:
                if response.startswith("{"):
                    retJson = json.loads(response)
                    retJson = restoreVariablesInDict(retJson, self.varDict.getAllDict())
                    return InstructResult(
                        Response(respDict=retJson),
                        useTool=False,
                        logMessage=logMessage)
                else:
                    return InstructResult(
                        Response(respStr=response),
                        useTool=False,
                        logMessage=logMessage)
            return InstructResult(
                Response(respStr=response), 
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        elif type(response) == Response:
            #return of a function
            if response.respList != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto,
                    logMessage=logMessage)
            elif response.respStr != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto,
                    logMessage=logMessage)
            elif response.respInt != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto,
                    logMessage=logMessage)
            else:
                raise RuntimeError(f"unsupported response[{response}]")
        elif type(response) == int:
            #return of a naive type
            return InstructResult(
                response=Response(respInt=response),
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        elif type(response) == float:
            #return of a naive type
            return InstructResult(
                response=Response(respFloat=response),
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        elif type(response) == bool:
            #return of a naive type
            return InstructResult(
                response=Response(respBool=response),
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        elif type(response) == list:
            return InstructResult(
                response=Response(respList=response),
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        elif type(response) == dict:
            return InstructResult(
                response=Response(respDict=response),
                useTool=useTool,
                goto=goto,
                logMessage=logMessage)
        else:
            raise RuntimeError(f"unsupported response type[{type(response)}]")

    def _formatCurInstruct(self, args :dict):
        allArgs = addDict(args, self.varDict.getAllDict())

        #call functions
        curInstruct = self.curInstruct
        if len(self.syntaxParser.funcsToCall) > 0:
            for funcBlock in self.syntaxParser.funcsToCall:
                resp = funcBlock.execute(
                    context=self.llmBlock.context, 
                    query=None, 
                    args=allArgs,
                    curInstruction=self)
                if curInstruct.strip() == funcBlock.getFuncCallPattern().strip():
                    self.onlyFuncCall = True
                    self.formattedInstruct = resp
                    return

                try:
                    curInstruct = curInstruct.replace(funcBlock.getFuncCallPattern(), str(resp.resp))
                except Exception as e:
                    raise RuntimeError(f"fail replace func[{funcBlock.getFuncCallPattern()}] with [{resp.resp}]")

        #restore variables
        try:
            self.formattedInstruct = restoreVariablesInStr(
                curInstruct, 
                allArgs)
        except Exception as e:
            raise RuntimeError(f"fail restore variables in [{curInstruct}]")
            
        if self.syntaxParser.flag == SyntaxParser.Flag.NONE:
            naiveType = Instruction._isNaiveType(self.formattedInstruct)
            if naiveType != None:
                self.isNaiveType = True
                self.formattedInstruct = naiveType 
                return
            
        if self.syntaxParser.flag == SyntaxParser.Flag.GOTO:
            self.formattedInstruct = self.formattedInstruct.replace(f"#GOTO {self.syntaxParser.label}", "")

    @staticmethod   
    def _isNaiveType(curInstruct: str):
        curInstruct = curInstruct.strip()
        
        if (curInstruct.startswith('"') and curInstruct.endswith('"')) or \
           (curInstruct.startswith("'") and curInstruct.endswith("'")):
            return curInstruct[1:-1]
        
        try:
            return int(curInstruct)
        except ValueError:
            pass
        
        try:
            return float(curInstruct)
        except ValueError:
            pass
        
        if curInstruct.lower() in {'True', 'False'}:
            return curInstruct.lower() == 'True'
        
        return None

    def _getCurLLM(self) -> EnhancedLLM:
        return self.llm if self.llm else self.syntaxParser.model

    def _getToolkit(self) -> Toolkit:
        return self.toolkit if self.toolkit else self.llmBlock.toolkit

class InstructionRecord:
    def __init__(self, instruction :Instruction, result :InstructResult) -> None:
        self.instruction = instruction
        self.result = result

class TaskEngine:
    def __init__(
            self, 
            llmBlock :LLMBlock,
            task :str) -> None:
        self.llmBlock = llmBlock
        self.task = task
        self.instructions :list[Instruction] = []
        self.lastInstruction :Instruction = None

        self.context = None

    def execute(
            self, 
            context: Context,
            taskArgs :dict, 
            instructions: list[Instruction], 
            execNodeParent: ExecNode,
            **kwargs) ->tuple:
        self.context = context
        self.lastInstruction = None
        self.instructions = instructions

        curIdx = 0
        instructResult :InstructResult = None  # 初始化 instructResult
        execNode :ExecNode = execNodeParent
        while curIdx < len(self.instructions):
            curInstruction = self.instructions[curIdx]
            if len(curInstruction.curInstruct.strip()) == 0 and \
                    curInstruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break

            execNode = ExecNodeLLM.build(
                execNodeParent=execNode, 
                instructionId=curInstruction.getId)
            
            stdout(f"\n{curInstruction.label} -> ", **kwargs)

            curInstruction.syntaxParser.reset()

            instructResult = self._step(
                    instruction=curInstruction, 
                    args=taskArgs, 
                    **kwargs)
            if instructResult.isRet():
                logger.info("end of the task")
                break

            #set variables
            if curInstruction.syntaxParser.flag == SyntaxParser.Flag.THOUGHT:
                self.llmBlock.setVarDictGlobal(KeyVarDictThtTask, curInstruction.formattedInstruct[:MaxLenThtTask])
                self.llmBlock.setVarDictGlobal(KeyVarDictThought, instructResult.response.resp)
            
            #process output
            curInstruction.syntaxParser.getInstrOutput().processOutputAndStore(
                output=instructResult.response.resp,
                stepLLMExtractor=self.llmBlock.stepLLMExtractor,
                varDict=curInstruction.varDict,
                toolkit=self.llmBlock.toolkit,
                retry=False,
                contextLen=curInstruction.syntaxParser.model.getContextWindow())
            if curInstruction.syntaxParser.getInstrOutput().hasError():
                raise RuntimeError(f"fail process output[{instructResult.response}]")

            if curInstruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break

            # handle goto
            if instructResult.isNext():
                curIdx += 1
            elif instructResult.goto:
                curIdx = self.getInstrLabels().index(instructResult.goto)
            elif curInstruction.syntaxParser.flag == SyntaxParser.Flag.GOTO:
                curIdx = self.getInstrLabels().index(curInstruction.syntaxParser.label)
            elif curInstruction.syntaxParser.flag == SyntaxParser.Flag.IF:
                curIdx = self.getInstrLabels().index(instructResult.response.resp)
            else:
                curIdx += 1

        return (True, instructResult.response)

    def getInstrLabels(self) -> list[str]:
        return [instruction.label for instruction in self.instructions]

    def _step(
            self, 
            instruction: Instruction, 
            args: dict, 
            **kwargs) -> InstructResult:
        t0 = time.time()
        instructResult = instruction.execute(args, **kwargs)
        INFO(logger, instructResult.logMessage + f" costSec[{time.time() - t0:.2f}]")
        return instructResult

class LLMBlock(BaseBlock):

    def __init__(
            self,
            agentName: str,
            context: Context = None,
            config: str = None,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> None:
        super().__init__(
            agentName=agentName,
            context=context, 
            config=config, 
            toolkit=toolkit, 
            repoFuncs=repoFuncs)

        self.systemPrompt = Loader.load(self.context.globalContext.settings.llmBasicConfig.systemPrompt)

        if taskExpr:
            self.task = taskExpr    
        else:
            self.task = Loader.load(task) if task else None

        self.jsonKeys = jsonKeys
        self.decomposeTask = decomposeTask
        self.taskEngine = TaskEngine(self, self.task)
        self.instructions = []
        self.stepLLMExtractor = StepLLMExtractor(
            globalContext=self.context.globalContext)
        self.globalSkillset = None

    def compile(self):
        if self.isCompiled:
            return

        if self.decomposeTask:
            self.instructions = self._decomposeTask(self.task)
            instrSummary = ""
            for instruction in self.instructions:
                instrSummary += f"{instruction}\n"
        else:
            self.instructions = [Instruction(
                llmBlock=self,
                curInstruct=self.task,
                label="1",
                prev=None)]

        self.isCompiled = True  # 设置编译标志为 True

    def execute(
            self, 
            context: Context,
            query: str = None, 
            args: dict = {},
            prevBlock: LLMBlock = None,
            execNodeParent: ExecNode = None,
            **kwargs) -> Response:
        super().execute(
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock, 
            execNodeParent=execNodeParent,
            **kwargs)

        INFO(logger, f"LLMBlock execute: task[{self.task[:10]}] query[{query}] args[{args}]")

        taskArgs = {**args}
        taskArgs["query"] = query
        self.context.varDict.remove("query")

        if not self.isCompiled:
            self.compile()  # 只在未编译时进行编译

        if self.getEnv():
            self.globalSkillset = self.getEnv().globalSkillset

        _, result = self.taskEngine.execute(
            context=context,
            taskArgs=taskArgs, 
            instructions=self.instructions,
            execNodeParent=execNodeParent,
            **kwargs)
        return result
    
    def recompile(self):
        self.isCompiled = False
        self.compile()

    def _decomposeTask(self, task: str) -> list[Instruction]:
        labelPattern = r'\s*([\w\u4e00-\u9fff]+[.、])\s+'
        
        lines = codeToLines(task)
        instructions = []
        currentLabel = None
        currentContent = []

        for line in lines:
            if not line:
                continue

            match = re.match(labelPattern, line)
            if match:
                if currentLabel is not None:
                    instructions.append((currentLabel, '\n'.join(currentContent)))
                
                currentLabel = match.group(1)[:-1]
                currentContent = [line[len(match.group(0)):].strip()]
            else:
                stripped = line.rstrip()
                if len(stripped) > 0:
                    currentContent.append(stripped)

        if currentLabel is not None:
            instructions.append((currentLabel, '\n'.join(currentContent)))

        if not instructions:
            instructions.append(("1", task.strip()))

        # 处理 Markdown 无序列表项
        processedInstructions = self._processMarkdownList(instructions)

        return [
            Instruction(
                llmBlock=self,
                curInstruct=instruction,
                label=label,
                prev=None)
            for label, instruction in processedInstructions
        ]

    def _processMarkdownList(self, instructions: list[tuple[str, str]]) -> list[tuple[str, str]]:
        processedInstructions = []
        lastNonDashLabel = None
        lastNonDashInstruction = None
        instructionsToRemove = {}
        for label, instruction in instructions:
            if label == "-":
                if lastNonDashInstruction:
                    instruction = f"{lastNonDashInstruction}. {instruction}"
                    instructionsToRemove[(lastNonDashLabel, lastNonDashInstruction)] = True
            else:
                lastNonDashLabel = label
                lastNonDashInstruction = instruction
            processedInstructions.append((label, instruction))

        for label, instruction in instructionsToRemove:
            instructions.remove((label, instruction))
        return processedInstructions

    @staticmethod
    def create(
            agentName: str,
            context: Context = None,
            config: str = None,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> 'LLMBlock':
        return LLMBlock(
            agentName=agentName,
            context=context,
            config=config,
            task=task,
            taskExpr=taskExpr,
            toolkit=toolkit,
            jsonKeys=jsonKeys,
            decomposeTask=decomposeTask,
            repoFuncs=repoFuncs)

if __name__ == "__main__":
    task = """
    Dog. #CODE 生成两个随机数 => [第个随机数，第二个随机数] -> random_nums
    Cat. #IF 如果{random_nums.0}是奇数，返回Tiger，如果是偶数，返回Monkey
    Tiger、 根据{random_nums.0}讲个笑话, #GOTO Slime -> joke
    Monkey、 说一首诗 -> poetry
    Fish. #RET {poetry}
    Slime. #RET {{"result": "{joke}"}}
    """
    llmBlock = LLMBlock.create(
        agentName="test", 
        taskExpr=task)
    print(llmBlock.execute().resp)
