from __future__ import annotations
from enum import Enum

import json
import logging
import re
from typing import Any, List, Optional, Tuple

from milkie.agent.base_block import BaseBlock
from milkie.agent.llm_block.step_llm_toolcall import StepLLMToolCall
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.config.constant import *
from milkie.context import Context
from milkie.functions.toolkits.toolbox import Toolbox
from milkie.functions.toolkits.toolkit import Toolkit, FuncExecRecord
from milkie.functions.toolkits.basic_toolkit import BasicToolkit
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, DEBUG
from milkie.response import Response
from milkie.trace import stdout
from milkie.utils.commons import mergeDict
from milkie.utils.data_utils import restoreVariablesInDict, restoreVariablesInStr

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
                你当前的指令是： {self.origInstruction}
                """

                resultPrompt += "前序指令情况如下:\n"
                resultPrompt += "```\n"
                resultPrompt += prevSummary
                resultPrompt += "```\n"

        resultPrompt += f"""
        你当前的指令是： {self.formattedInstruction}
        """
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

            任务目标: [{llmBlock.getVarDict()[KeyVarDictThtTask]}]
            任务思路：[{llmBlock.getVarDict()[KeyVarDictThought]}]
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
        super().__init__(llmBlock.context.globalContext, promptMaker)
        self.llmBlock = llmBlock

class StepLLMInstrAnalysis(StepLLMBlock):
    def __init__(
            self, 
            promptMaker :PromptMakerInstruction, 
            llmBlock :LLMBlock, 
            instruction: Instruction) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instruction = instruction
        self.instructionRecords = llmBlock.taskEngine.instructionRecords
        self.stepToolCall = StepLLMToolCall(llmBlock.context.globalContext)
        
    def makePrompt(self, useTool: bool = False, **args) -> str:
        if args.get("prompt", None) == InstFlagDecompose:
            return self.promptMaker.promptForDecompose(self.llmBlock)
        elif args.get("prompt", None) == InstFlagThought:
            return self.promptMaker.promptForThought(self.llmBlock)

        result = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)

        if self.instruction.syntaxParser.getNormalFormat():
            result += f"""
            请按照下述语义严格以 jsonify 格式输出结果：{self.instruction.syntaxParser.getOutputSyntax()}，现在请直接输出 json:
            """
        elif not useTool:
            result += f"""
            请直接输出结果，不要输出任何其他内容:
            """
        return result

    def formatResult(self, result :Response, **kwargs) -> InstAnalysisResult:
        allDict = self.llmBlock.getVarDict().getAllDict()
        needToParse = self.instruction.formattedInstruct.find("{") >= 0
        toolUsed, result = StepLLMInstrAnalysis.readFromGen(
            result,
            stepToolCall=self.stepToolCall,
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
        if self.llmBlock.toolkit:
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

    @staticmethod
    def readFromGen(
            response :Response, 
            stepToolCall: Optional[StepLLMToolCall] = None, 
            respToolbox: Optional[Toolbox] = None,
            **kwargs) -> Tuple[Optional[str], str]:
        toolUsed = None
        result = []
        currentSentence = []

        def processDeltaContentNormal(deltaContent: str) -> str:
            nonlocal currentSentence
            currentSentence.append(deltaContent)
            if any(symbol in deltaContent for symbol in SymbolEndSentence):
                sentence = ''.join(currentSentence)
                if stepToolCall:
                    toolCheck = stepToolCall.completionAndFormat(
                        query_str=sentence,
                        tools=respToolbox
                    )

                    if toolCheck["need_tool"] and respToolbox:
                        stdout(f"\nexecute call {toolCheck['tool_name']} with args {toolCheck['tool_args']} ==> ", **kwargs)
                        respToolbox.execFromJson(
                            funcName=toolCheck["tool_name"], 
                            args=toolCheck["tool_args"],
                            allDict={},
                            **{KeywordMute: True},
                            **kwargs)
                        stdout(f"\n===============", **kwargs)
                currentSentence = []
            return deltaContent
            
        for chunk in response.respGen:
            if chunk.raw.role == "assistant":
                continue

            if toolUsed is None:
                if chunk.raw.tool_calls:
                    toolUsed = chunk.raw.tool_calls[0].function.name
                    continue
                toolUsed = ""
            
            if toolUsed == "":
                deltaContent = processDeltaContentNormal(chunk.raw.content)
            else:
                if chunk.raw.content is None:
                    deltaContent = chunk.raw.tool_calls[0].function.arguments
                else:
                    deltaContent = ""

            result.append(deltaContent)
            stdout(deltaContent, end="", flush=True, **kwargs)

        return (toolUsed or None, ''.join(result))

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
            curInstruct, 
            llmBlock.repoFuncs, 
            llmBlock.getEnv().getGlobalToolkits())
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

    def execute(self, args :dict, **kwargs) -> InstructResult:
        try:
            self._formatCurInstruct(args)
        except Exception as e:
            raise RuntimeError(f"fail parse instruct[{self.curInstruct}]: {str(e)}")

        if self.onlyFuncCall or self.isNaiveType:
            # in this case, the response is the result of the only function call, or a naive type
            stdout(self.formattedInstruct, args=args, **kwargs)
            return self._createResult(
                self.formattedInstruct,
                useTool=False,
                goto=None,
                analysis=self.curInstruct,
                logType="naive")

        useTool = (self.llmBlock.toolkit != None)
        logTypes = []
        if self.syntaxParser.flag == SyntaxParser.Flag.CODE:
            result = self.llmBlock.toolkit.genCodeAndRun(self.formattedInstruct, self.varDict.getAllDict())
            stdout(result, args=args, **kwargs)
            return self._createResult(
                result,
                useTool=True,
                goto=None,
                analysis=self.curInstruct,
                logType="code")

        elif self.syntaxParser.flag == SyntaxParser.Flag.IF:
            result = self.llmBlock.toolkit.genCodeAndRun(self.formattedInstruct, self.varDict.getAllDict())
            stdout(result, args=args, **kwargs)
            return self._createResult(
                result,
                useTool=True,
                goto=result,
                analysis=self.curInstruct,
                logType="if")

        elif self.syntaxParser.flag == SyntaxParser.Flag.PY:
            def preprocessPyInstruct(instruct: str):
                instruct = instruct.replace("$varDict", "self.varDict")
                instruct = instruct.replace(KeyNext, f'"{KeyNext}"')
                instruct = instruct.replace(KeyRet, f'"{KeyRet}"')
                return instruct

            result = self.llmBlock.toolkit.runCode(
                preprocessPyInstruct(self.formattedInstruct),
                self.varDict.getAllDict())
            stdout(result, args=args, **kwargs)
            return self._createResult(
                result,
                useTool=False,
                goto=None,
                analysis=self.curInstruct,
                logType="py")

        elif self.syntaxParser.flag == SyntaxParser.Flag.CALL:
            query = self.syntaxParser.callArg
            MaxRetry = 2
            for i in range(MaxRetry):
                stdout(f"execute call {self.syntaxParser.callObj} with query {query} ==> ", args=args, **kwargs)
                resp = self.llmBlock.context.getEnv().execute(
                    agentName=self.syntaxParser.callObj,
                    query=query,
                    args=args)

                instrOutput = self.syntaxParser.getInstrOutput()
                instrOutput.processOutputAndStore(
                    stepLLMExtractor=self.llmBlock.stepLLMExtractor,
                    output=resp.respStr,
                    varDict=self.varDict,
                    retry=True)
                if not instrOutput.hasError():
                    stdout(f"==> {resp.respStr}", args=args, **kwargs)
                    return self._createResult(
                        resp,
                        useTool=False,
                        goto=None,
                        analysis=self.curInstruct,
                        logType="call")
                
                query = "\n".join(instrOutput.getErrMsgs() + [query])
            raise RuntimeError(f"fail execute instruction[{self.curInstruct}] error[{instrOutput.getErrMsgs()}]")

        elif self.syntaxParser.flag == SyntaxParser.Flag.RET and self.syntaxParser.returnVal:
            stdout(self.formattedInstruct, args=args, **kwargs)
            return self._createResult(
                self.formattedInstruct,
                useTool=False,
                goto=None,
                analysis=self.curInstruct,
                logType="ret")

        if self.syntaxParser.flag == SyntaxParser.Flag.THOUGHT:
            self.promptMaker.promptForThought(self.llmBlock)
            args["prompt"] = InstFlagThought
            useTool = False
            logTypes.append("thought")

        elif self.syntaxParser.flag == SyntaxParser.Flag.DECOMPOSE:
            self.promptMaker.promptForDecompose(self.llmBlock)
            args["prompt"] = InstFlagDecompose
            useTool = False
            logTypes.append("decompose")

        else:
            args["prompt"] = None

        kwargs = {}
        if not self.syntaxParser.respToolbox and useTool:
            kwargs["tools"] = self.llmBlock.toolkit
        
        if self.syntaxParser.respToolbox:
            kwargs["respToolbox"] = self.syntaxParser.respToolbox
        
        instAnalysisResult = self.stepInstAnalysis.streamAndFormat(
            args=args,
            **kwargs)

        if instAnalysisResult.result == InstAnalysisResult.Result.ANSWER:
            logTypes.append("ans")
            return self._createResult(
                instAnalysisResult.response, 
                useTool=False,
                goto=None,
                analysis=instAnalysisResult.response,
                logType="/".join(logTypes))

        return self._createResult(
            instAnalysisResult.funcExecRecords[0].result,
            useTool=True,
            goto=None,
            analysis=instAnalysisResult.response,
            answer=instAnalysisResult.funcExecRecords[0].result,
            logType="tool",
            toolName=instAnalysisResult.funcExecRecords[0].tool.get_function_name())
        
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
            if response.respList:
                return InstructResult(
                    response=response,
                    useTool=useTool,
                    goto=goto)
            elif response.respStr:
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
        else:
            raise RuntimeError(f"unsupported response type[{type(response)}]")

    def _formatCurInstruct(self, args :dict):
        allArgs = mergeDict(args, self.varDict.getAllDict())

        #call functions
        curInstruct = self.curInstruct
        if len(self.syntaxParser.funcsToCall) > 0:
            for funcBlock in self.syntaxParser.funcsToCall:
                resp = funcBlock.execute(args=allArgs)
                if curInstruct.strip() == funcBlock.getFuncPattern().strip():
                    self.onlyFuncCall = True
                    self.formattedInstruct = resp
                    return

                try:
                    curInstruct = curInstruct.replace(funcBlock.getFuncPattern(), str(resp.resp))
                except Exception as e:
                    raise RuntimeError(f"fail replace func[{funcBlock.getFuncPattern()}] with [{resp.resp}]")
            
        if self.syntaxParser.flag == SyntaxParser.Flag.NONE:
            naiveType = Instruction._isNaiveType(curInstruct)
            if naiveType != None:
                self.isNaiveType = True
                self.formattedInstruct = naiveType 
                return
            
        #restore variables
        try:
            self.formattedInstruct = restoreVariablesInStr(
                curInstruct, 
                allArgs)
        except Exception as e:
            raise RuntimeError(f"fail restore variables in [{curInstruct}]")

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

    def execute(
            self, 
            taskArgs :dict, 
            instructions: list[tuple[str, Instruction]], 
            **kwargs) -> tuple:
        self.lastInstruction = None
        self.instructions = instructions

        curIdx = 0
        instructResult :InstructResult = None  # 初始化 instructResult
        while curIdx < len(self.instructions):
            label, instruction = self.instructions[curIdx]
            if len(instruction.curInstruct.strip()) == 0 and \
                    instruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break
            
            stdout(f"{label} -> ", **kwargs)
            instructResult = self._step(instruction, args=taskArgs, **kwargs)
            self.llmBlock.setResp(label, instructResult.response.resp)
            if instructResult.isRet():
                logger.info("end of the task")
                break

            #set variables
            if instruction.syntaxParser.flag == SyntaxParser.Flag.THOUGHT:
                self.llmBlock.setVarDictGlobal(KeyVarDictThtTask, instruction.formattedInstruct[:MaxLenThtTask])
                self.llmBlock.setVarDictGlobal(KeyVarDictThought, instructResult.response.resp)
            
            #process output
            instruction.syntaxParser.getInstrOutput().processOutputAndStore(
                stepLLMExtractor=self.llmBlock.stepLLMExtractor,
                output=instructResult.response.respStr,
                varDict=instruction.varDict,
                retry=False)
            if instruction.syntaxParser.getInstrOutput().hasError():
                raise RuntimeError(f"fail process output[{instructResult.response}]")

            #append instruction record
            self.instructionRecords.append(
                InstructionRecord(
                    instruction=instruction, 
                    result=instructResult))

            if instruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break

            # adjust instructions and curIdx
            if instruction.syntaxParser.flag == SyntaxParser.Flag.DECOMPOSE:
                newInstructions = self.llmBlock._decomposeTask(instructResult.response.resp)
                for i, (label, instruction) in enumerate(newInstructions):
                    self.instructions.insert(curIdx + i + 1, (label, instruction))  
                curIdx += 1
                continue

            # handle goto
            if instructResult.isNext():
                curIdx += 1
            elif instructResult.goto:
                curIdx = self.getInstrLabels().index(instructResult.goto)
            elif instruction.syntaxParser.flag == SyntaxParser.Flag.GOTO:
                curIdx = self.getInstrLabels().index(instruction.syntaxParser.label)
            else:
                curIdx += 1

        return (True, instructResult.response)

    def getInstrLabels(self) -> list[str]:
        return [label for label, _ in self.instructions]

    def _step(self, instruction: Instruction, args: dict, **kwargs) -> InstructResult:
        instructResult = instruction.execute(args, **kwargs)
        return instructResult

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

        self.systemPrompt = self.context.globalContext.globalConfig.getLLMConfig().systemPrompt

        if taskExpr:
            self.task = taskExpr    
        else:
            self.task = Loader.load(task) if task else None

        self.jsonKeys = jsonKeys
        self.decomposeTask = decomposeTask

        self.taskEngine = TaskEngine(self, self.task)
        self.instructions = []
        self.isCompiled = False
        self.stepLLMExtractor = StepLLMExtractor(
            globalContext=self.context.globalContext)

    def compile(self):
        if self.isCompiled:
            return  # 如果已经编译过，直接返回

        if self.decomposeTask:
            self.instructions = self._decomposeTask(self.task)
            instrSummary = ""
            for label, instruction in self.instructions:
                instrSummary += f"{label}: {instruction}\n"
            INFO(logger, f"analysis ans[{instrSummary}]")
        else:
            self.instructions = [("1", Instruction(
                llmBlock=self,
                curInstruct=self.task,
                label="1",
                prev=None))]

        self.isCompiled = True  # 设置编译标志为 True

    def execute(
            self, 
            query: str = None, 
            args: dict = {},
            prevBlock: LLMBlock = None,
            **kwargs) -> Response:
        self.updateFromPrevBlock(prevBlock, args)

        INFO(logger, f"LLMBlock execute: query[{query}] args[{args}]")

        taskArgs = {"query_str": query, **args}
        if not self.isCompiled:
            self.compile()  # 只在未编译时进行编译
        _, result = self.taskEngine.execute(
            taskArgs=taskArgs, 
            instructions=self.instructions,
            **kwargs)
        return result
    
    def recompile(self):
        self.isCompiled = False
        self.compile()

    def _decomposeTask(self, task: str) -> list[tuple[str, Instruction]]:
        labelPattern = r'\s*([\w\u4e00-\u9fff]+[.、]|\-)\s+'
        
        lines = task.split('\n')
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
                
                currentLabel = "-" if match.group(1) == "-" else match.group(1)[:-1]
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
        return LLMBlock(context, config, usePrevResult, task, taskExpr, toolkit, jsonKeys, decomposeTask, repoFuncs)

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
