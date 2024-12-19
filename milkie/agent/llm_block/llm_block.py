from __future__ import annotations
from enum import Enum

import json
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from milkie.agent.base_block import BaseBlock
from milkie.agent.llm_block.step_llm_toolcall import StepLLMToolCall
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.config.constant import *
from milkie.context import Context
from milkie.functions.toolkits.toolbox import Toolbox
from milkie.functions.toolkits.toolkit import Toolkit, FuncExecRecord
from milkie.functions.toolkits.basic_toolkit import BasicToolkit
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, DEBUG
from milkie.response import Response
from milkie.trace import stdout
from milkie.utils.commons import mergeDict
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

class PromptMakerInstruction(PromptMaker):

    def __init__(
            self, 
            toolkit :Toolkit, 
            usePrevResult :bool) -> None:
        super().__init__(toolkit)

        self.usePrevResult = usePrevResult
        self.origInstruction :str = None
        self.formattedInstruction :str = None
        self.instructionDetails :str = None
        self.optionDecompose = False
        self.prev = None
    
    def setOrigInstruction(self, instruction: str):
        self.origInstruction = instruction
        self.formattedInstruction = instruction

    def setFormattedInstruction(self, instruction: str):
        self.formattedInstruction = instruction

    def setInstructionDetails(self, details: str):
        self.instructionDetails = details

    def setPrev(self, prev):
        self.prev = prev

    def promptForInstruction(
            self, 
            instructionRecords :list[InstructionRecord],
            **args):
        resultPrompt = ""
        if self.usePrevResult:
            prevSummary = self.prevStepSummary(instructionRecords)
            if len(prevSummary) > 0:
                resultPrompt += f"""
                你当前的Query是： {self.origInstruction}
                """

                resultPrompt += "前序Query情况如下:\n"
                resultPrompt += "```\n"
                resultPrompt += prevSummary
                resultPrompt += "```\n"

        resultPrompt += f"""{self.formattedInstruction}"""
        return resultPrompt

    def promptForThought(self, llmBlock :LLMBlock):
        result = f"""
            任务目标: [{self.formattedInstruction}]
            Toolkit: {llmBlock.toolkit.getToolsDesc()}
            请思考如何解决这个问题，基于Toolkit中的工具。
            方法如下：
            """
        return result

    def promptForDecompose(self, llmBlock :LLMBlock):
        sampleToolKit = BasicToolkit(llmBlock.context.globalContext)
        sampleEmailDesc = Toolkit.getToolsDescWithSingleFunc(sampleToolKit.sendEmail)
        result = f"""
            请将任务分解为指令，要求如下：
            (1)每条指令必须仅对应一次Toolkit工具调用，或者直接生成response
            (2)这些指令的逐步执行可以完成任务目标
            (3)生成的指令的格式请参考下面示例
            
            示例如下

            ```
            任务目标：针对 {{topic}} 写一篇文章，做摘要，用 markdown 格式格式化，并且邮件发送给{{email}}
            Toolkit：{sampleEmailDesc}
            任务分解：
            1. 详细分析下面的问题{{topic}} -> topicInfo
            2. 我现在要针对主题{{topic}}写一篇文章，根据下述信息写一篇摘要: --{{topicInfo}}-- -> summary
            3. 用 markdown 格式化下面内容--{{summary}}-- -> markdown
            4. 邮件发送给{{email}}, 邮件标题为{{title}}, 邮件内容为{{markdown}}
            ```

            任务目标: [{llmBlock.getVarDict().get(KeyVarDictThtTask)}]
            任务思路：[{llmBlock.getVarDict().get(KeyVarDictThought)}]
            Toolkit：{llmBlock.toolkit.getToolsDesc()}
            注意：分解任务时，请不要对已知的信息做重复计算
            任务分解：
            """
        return result

    def prevStepSummary(self, instructionRecords :list[InstructionRecord]):
        result = ""
        if len(instructionRecords) > 0:
            result += f"上一步总结: {instructionRecords[-1].synthesizedInstructionRecord[:MaxLenLastStepResult]}\n"
            result += f"上一步详情: {str(instructionRecords[-1].result.response.resp)[:MaxLenLastStepResult]}\n"
        return result

class StepLLMBlock(StepLLM):
    def __init__(self, promptMaker, llmBlock :LLMBlock):
        super().__init__(
            globalContext=llmBlock.context.globalContext, 
            promptMaker=promptMaker,
            llm=llmBlock.context.globalContext.settings.llmDefault)

        self.llmBlock = llmBlock

class StepLLMInstrAnalysis(StepLLMBlock):
    def __init__(
            self, 
            promptMaker :PromptMakerInstruction, 
            llmBlock :LLMBlock, 
            instruction: Instruction) -> None:
        super().__init__(
            promptMaker=promptMaker, 
            llmBlock=llmBlock)

        self.instruction = instruction
        self.instructionRecords = llmBlock.taskEngine.instructionRecords
        self.stepToolCall = StepLLMToolCall(llmBlock.context.globalContext)

    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        systemPrompt = super().makeSystemPrompt(args=args, **kwargs)
        if "experts" in kwargs:
            systemPrompt += f"""
            注意：专家团成员如下
            ```
            """
            for name, agent in kwargs["experts"].items():
                systemPrompt += f"{name} -> {agent.desc}\n"
            systemPrompt += '''```\n
            如果需要咨询专家团，请使用 ” @expertName ('->'前面的字符串) + 问题 + ？“ 来调用专家团中的专家。
            '''
        return systemPrompt
        
    def makePrompt(self, useTool: bool = False, **args) -> str:
        if args.get("prompt_flag", None) == InstFlagDecompose:
            return self.promptMaker.promptForDecompose(self.llmBlock)
        elif args.get("prompt_flag", None) == InstFlagThought:
            return self.promptMaker.promptForThought(self.llmBlock)

        result = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)

        if self.instruction.syntaxParser.getNormalFormat():
            result += f"""
            请按照下述语义严格以 jsonify 格式输出结果：{self.instruction.syntaxParser.getJsonOutputSyntax()}，现在请直接输出 json:
            """
        elif not useTool:
            result += f"""
            请直接输出结果，不要输出任何其他内容:
            """
        return result

    def formatResult(self, result :Response, **kwargs) -> InstAnalysisResult:
        allDict = self.llmBlock.getVarDict().getAllDict()
        needToParse = self.instruction.formattedInstruct.find("{") >= 0

        streamingProcessor = StreamingProcessor(
            respToolbox=self.llmBlock.respToolbox,
            context=self.llmBlock.context)
        toolUsed, result = streamingProcessor.readFromGen(
            response=result,
            **kwargs)
        if toolUsed:
            funcExecRecords = self.llmBlock.toolkit.exec(
                [(toolUsed, result)], 
                allDict,
                needToParse=needToParse)
            stdout(funcExecRecords[0].result, **kwargs)
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=f"{toolUsed}({result})")
            
        # if function call is not in tools, but it is an answer
        if not self.llmBlock.toolkit.isEmpty():
            funcExecRecords = self.llmBlock.toolkit.extractToolFromMsg(
                result,
                allDict,
                needToParse=needToParse)
            if funcExecRecords:
                stdout(funcExecRecords[0].result, **kwargs)
                return InstAnalysisResult(
                    InstAnalysisResult.Result.TOOL,
                    funcExecRecords=funcExecRecords,
                    response=result)

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result)

class StreamingProcessor:
    def __init__(self,
                 respToolbox: Toolbox,
                 context: Context):
        self.respToolbox = respToolbox
        self.context = context
        self.currentContent = []
        self.currentSentence = []
        self.currentTools = []

    def readFromGen(
            self,
            response: Response, 
            **kwargs) -> Tuple[Optional[str], str]:
        expertsResp = None

        try:
            if not response.respGen:
                logger.warning("No response generator found")
                return None, ""

            for chunk in response.respGen:
                content = None
                if hasattr(chunk.raw, "content") and chunk.raw.content is not None:
                    content = chunk.raw.content
                elif hasattr(chunk.raw, "delta") and hasattr(chunk.raw.delta, "content"):
                    content = chunk.raw.delta.content
                elif hasattr(chunk.raw, "tool_calls") and len(chunk.raw.tool_calls) > 0:
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

                if content:
                    expertsResp = self.processDeltaContent(
                        deltaContent=content,
                        **kwargs)

            stdout("", info=True, flush=True, **kwargs)

            if self.currentSentence:
                lastSentence = "".join(self.currentSentence)
                self.currentContent.append(lastSentence)
                if self.respToolbox and lastSentence.startswith("@"):
                    expertsResp = self.respToolbox.queryExpert(lastSentence, context=self.context)
                    self.currentContent.append(expertsResp)

        except Exception as e:
            logger.error(f"Error in readFromGen: {e}", exc_info=True)
            raise

        if len(self.currentTools) > 0:
            return self.currentTools[0]["name"], self.currentTools[0]["args"]
        else:
            return None, "".join(self.currentContent)

    def processDeltaContent(
            self,
            deltaContent: str,
            **kwargs) -> str:
        endIndex = -1
        for i, char in enumerate(deltaContent):
            if char in SymbolEndSentence:
                endIndex = i
                break
        
        expertsResp = ""
        if endIndex >= 0:
            currentPart = deltaContent[:endIndex+1]
            self.currentSentence.append(currentPart)
            stdout(currentPart, info=True, end="", flush=True, **kwargs)
            
            expertsResp = self.processExpertQuery()
            self.currentContent.append(expertsResp)
            
            self.currentSentence = [deltaContent[endIndex+1:]]
            stdout(deltaContent[endIndex+1:], info=True, end="", flush=True, **kwargs)
        else:
            self.currentSentence.append(deltaContent)
            stdout(deltaContent, info=True, end="", flush=True, **kwargs)
        
        return expertsResp

    def processExpertQuery(self) -> str:
        sentence = "".join(self.currentSentence)
        self.currentContent.append(sentence)
        
        if self.respToolbox and sentence.startswith("@"):
            expertsResp = self.respToolbox.queryExpert(sentence, context=self.context)
            return expertsResp
        return ""

class StepLLMSynthsisInstructionRecord(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock, instructionRecord: InstructionRecord) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instructionRecord = instructionRecord

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs) -> str:
        resultPrompt = f"""
        指令为：
        {self.instructionRecord.instruction.curInstruct}

        指令执行结果为：
        {self.instructionRecord.result.response.resp}

        请将指令指令本身和执行结果总结为一句话，请直接给出总结结果，总结结果为：
        """
        return resultPrompt

    def formatResult(self, result :Response):
        return result.resp

class InstructResult:
    def __init__(
            self, 
            response :Response,
            goto :str = None,
            useTool :bool = False):
        self.response = response
        self.goto = goto
        self.useTool = useTool

    def isRet(self):
        return self.response.respStr == KeyRet

    def isNext(self):
        return self.response.respStr == KeyNext

from milkie.agent.llm_block.syntax_parser import SyntaxParser

class Instruction:
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            curInstruct: str,
            label: str = None,
            observation: str = None,
            prev = None) -> None:
        self.llmBlock = llmBlock
        self.syntaxParser = SyntaxParser(
            label=label,
            settings=llmBlock.context.globalContext.settings,
            instruction=curInstruct, 
            repoFuncs=llmBlock.repoFuncs, 
            toolkits=llmBlock.getEnv().getGlobalToolkits() if llmBlock.getEnv() else None)
        self.curInstruct = self.syntaxParser.getInstruction()
        self.formattedInstruct = self.curInstruct
        self.onlyFuncCall = False
        self.isNaiveType = False
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.instructionRecords = llmBlock.taskEngine.instructionRecords
        self.varDict = llmBlock.getVarDict()

        self.promptMaker = PromptMakerInstruction(
            toolkit=self.llmBlock.toolkit,
            usePrevResult=self.llmBlock.usePrevResult)
        self.promptMaker.setTask(llmBlock.taskEngine.task)
        self.promptMaker.setOrigInstruction(self.curInstruct)
        self.promptMaker.setPrev(self.prev)
        self.stepInstAnalysis = StepLLMInstrAnalysis(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock,
            instruction=self)
        self.llm :EnhancedLLM = None
        self.reasoning :Reasoning = None

    def execute(self, args :dict, **kwargs) -> InstructResult:
        try:
            self._formatCurInstruct(args)
        except Exception as e:
            raise RuntimeError(f"fail parse instruct[{self.curInstruct}]: {str(e)}")

        if self.onlyFuncCall or self.isNaiveType:
            # in this case, the response is the result of the only function call, or a naive type
            return self._processNaiveType(args, **kwargs)

        useTool = (self.llmBlock.toolkit != None)
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

        if self.syntaxParser.flag == SyntaxParser.Flag.THOUGHT:
            self.promptMaker.promptForThought(self.llmBlock)
            args["prompt_flag"] = InstFlagThought
            useTool = False
            logTypes.append("thought")

        elif self.syntaxParser.flag == SyntaxParser.Flag.DECOMPOSE:
            self.promptMaker.promptForDecompose(self.llmBlock)
            args["prompt_flag"] = InstFlagDecompose
            useTool = False
            logTypes.append("decompose")

        else:
            args["prompt_flag"] = None

        if useTool:
            kwargs["tools"] = self.llmBlock.toolkit.getTools()
        
        if self.llmBlock.getEnv():
            respToolbox = Toolbox(self.llmBlock.getEnv().getGlobalToolkits())
        else:
            respToolbox = None

        if self.syntaxParser.respToolbox:
            respToolbox.merge(self.syntaxParser.respToolbox)

        if self.llmBlock.respToolbox:
            respToolbox.merge(self.llmBlock.respToolbox)
        kwargs["respToolbox"] = respToolbox
        
        instAnalysisResult = self.stepInstAnalysis.streamAndFormat(
            llm=self.llm if self.llm else self.syntaxParser.model,
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
            return instruction.llmBlock.context.getEnv().execute(
                agentName=instruction.syntaxParser.callObj,
                query=theArgs["query"],
                args=theArgs["args"])
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
            stdout(f"execute call {self.syntaxParser.callObj} with query {query} ==> ", args=args, **kwargs)
                
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
                toolkit=self.llmBlock.toolkit,
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
        logMessage = f"instrExec({self.label}|{logType}): instr[{self.formattedInstruct}] "
        if logType == "tool":
            logMessage += f"tool[{logData.get('toolName', '')}] "
        logMessage += f"ans[{response}]"
        
        INFO(logger, logMessage)
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
                        useTool=False)
                else:
                    return InstructResult(
                        Response(respStr=response),
                        useTool=False)
            return InstructResult(
                Response(respStr=response), 
                useTool=useTool,
                goto=goto)
        elif type(response) == Response:
            #return of a function
            if response.respList != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto)
            elif response.respStr != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto)
            elif response.respInt != None:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto)
            else:
                raise RuntimeError(f"unsupported response[{response}]")
        elif type(response) == int:
            #return of a naive type
            return InstructResult(
                response=Response(respInt=response),
                useTool=useTool,
                goto=goto)
        elif type(response) == float:
            #return of a naive type
            return InstructResult(
                response=Response(respFloat=response),
                useTool=useTool,
                goto=goto)
        elif type(response) == bool:
            #return of a naive type
            return InstructResult(
                response=Response(respBool=response),
                useTool=useTool,
                goto=goto)
        elif type(response) == list:
            return InstructResult(
                response=Response(respList=response),
                useTool=useTool,
                goto=goto)
        elif type(response) == dict:
            return InstructResult(
                response=Response(respDict=response),
                useTool=useTool,
                goto=goto)
        else:
            raise RuntimeError(f"unsupported response type[{type(response)}]")

    def _formatCurInstruct(self, args :dict):
        allArgs = mergeDict(args, self.varDict.getAllDict())

        #call functions
        curInstruct = self.curInstruct
        if len(self.syntaxParser.funcsToCall) > 0:
            for funcBlock in self.syntaxParser.funcsToCall:
                resp = funcBlock.execute(
                    context=self.llmBlock.context, 
                    query=None, 
                    args=allArgs,
                    curInstruction=self)
                if curInstruct.strip() == funcBlock.getFuncPattern().strip():
                    self.onlyFuncCall = True
                    self.formattedInstruct = resp
                    return

                try:
                    curInstruct = curInstruct.replace(funcBlock.getFuncPattern(), str(resp.resp))
                except Exception as e:
                    raise RuntimeError(f"fail replace func[{funcBlock.getFuncPattern()}] with [{resp.resp}]")

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
        self.promptMaker.setFormattedInstruction(self.formattedInstruct)

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

class InstructionRecord:
    def __init__(self, instruction :Instruction, result :InstructResult) -> None:
        self.instruction = instruction
        self.result = result

        if instruction.llmBlock.usePrevResult:
            if result.useTool:
                self.stepSynthsisInstructionRecord = StepLLMSynthsisInstructionRecord(
                    promptMaker=instruction.promptMaker,
                    llmBlock=instruction.llmBlock,
                    instructionRecord=self)
                self.synthesizedInstructionRecord = self.stepSynthsisInstructionRecord.run()
            else:
                self.synthesizedInstructionRecord = self.result.response.resp

class TaskEngine:
    def __init__(
            self, 
            llmBlock :LLMBlock,
            task :str) -> None:
        self.llmBlock = llmBlock
        self.task = task
        self.instructions :list[tuple[str, Instruction]] = []
        self.lastInstruction :Instruction = None
        self.instructionRecords :list[InstructionRecord] = []

        self.context = None

    def execute(
            self, 
            context: Context,
            taskArgs :dict, 
            instructions: list[tuple[str, Instruction]], 
            **kwargs) ->tuple:
        self.context = context
        self.lastInstruction = None
        self.instructions = instructions

        curIdx = 0
        instructResult :InstructResult = None  # 初始化 instructResult
        while curIdx < len(self.instructions):
            curLabel, curInstruction = self.instructions[curIdx]
            if len(curInstruction.curInstruct.strip()) == 0 and \
                    curInstruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break
            
            stdout(f"\n{curLabel} -> ", **kwargs)

            curInstruction.syntaxParser.reset()

            instructResult = self._step(instruction=curInstruction, args=taskArgs, **kwargs)
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

            #append instruction record
            self.instructionRecords.append(
                InstructionRecord(
                    instruction=curInstruction, 
                    result=instructResult))

            if curInstruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break

            # adjust instructions and curIdx
            if curInstruction.syntaxParser.flag == SyntaxParser.Flag.DECOMPOSE:
                newInstructions = self.llmBlock._decomposeTask(instructResult.response.resp)
                for i, (curLabel, curInstruction) in enumerate(newInstructions):
                    self.instructions.insert(curIdx + i + 1, (curLabel, curInstruction))  
                curIdx += 1
                continue

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
        return [label for label, _ in self.instructions]

    def _step(self, instruction: Instruction, args: dict, **kwargs) -> InstructResult:
        return instruction.execute(args, **kwargs)

class LLMBlock(BaseBlock):

    def __init__(
            self,
            context: Context = None,
            config: str = None,
            usePrevResult: bool = DefaultUsePrevResult,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> None:
        super().__init__(context, config, toolkit, usePrevResult, repoFuncs)

        self.usePrevResult = usePrevResult
        self.systemPrompt = self.context.globalContext.settings.llmBasicConfig.systemPrompt

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
        self.respToolbox = None

    def compile(self):
        if self.isCompiled:
            return

        if self.decomposeTask:
            self.instructions = self._decomposeTask(self.task)
            instrSummary = ""
            for _, instruction in self.instructions:
                instrSummary += f"{instruction}\n"
        else:
            self.instructions = [("1", Instruction(
                llmBlock=self,
                curInstruct=self.task,
                label="1",
                prev=None))]

        self.isCompiled = True  # 设置编译标志为 True

    def execute(
            self, 
            context: Context,
            query: str = None, 
            args: dict = {},
            prevBlock: LLMBlock = None,
            **kwargs) -> Response:
        super().execute(
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)

        INFO(logger, f"LLMBlock execute: task[{self.task[:10]}] query[{query}] args[{args}]")

        taskArgs = {**args}
        taskArgs["query"] = query

        if not self.isCompiled:
            self.compile()  # 只在未编译时进行编译

        if self.getEnv():
            self.respToolbox = Toolbox.createToolbox(
                globalToolkits=self.getEnv().globalToolkits,
                toolkits=[agent.name for agent in self.getEnv().agents.values() if agent.name != "stdin"])

        _, result = self.taskEngine.execute(
            context=context,
            taskArgs=taskArgs, 
            instructions=self.instructions,
            **kwargs)
        return result
    
    def recompile(self):
        self.isCompiled = False
        self.compile()

    def _decomposeTask(self, task: str) -> list[tuple[str, Instruction]]:
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
            (label, Instruction(
                llmBlock=self,
                curInstruct=instruction,
                label=label,
                prev=None))
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
            context: Context = None,
            config: str = None,
            usePrevResult: bool = DefaultUsePrevResult,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> 'LLMBlock':
        return LLMBlock(
            context=context,
            config=config,
            usePrevResult=usePrevResult,
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
    llmBlock = LLMBlock.create(taskExpr=task)
    print(llmBlock.execute().resp)
