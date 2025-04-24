from __future__ import annotations
from enum import Enum

import json
import logging
import re
import time
from typing import Any, Callable
import uuid

from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNode, ExecNodeLLM, ExecNodeLabel, ExecNodeSequence, ExecNodeSkill, ExecNodeType
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.agent.step_llm_streaming import InstAnalysisResult, StepLLMStreaming
from milkie.config.constant import *
from milkie.context import Context
from milkie.functions.code_interpreter import CodeInterpreter
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.prompt.prompt import Loader
from milkie.log import INFO, DEBUG
from milkie.response import Response
from milkie.types.object_type import ObjectType, ObjectTypeFactory
from milkie.utils.commons import addDict
from milkie.utils.data_utils import codeToLines, preprocessPyCode, restoreVariablesInDict, restoreVariablesInStr

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

class StepLLMInstrAnalysis(StepLLMStreaming):
    def __init__(
            self, 
            instruction: Instruction) -> None:
        super().__init__()

        self.instruction = instruction
        self.needToParse = instruction.formattedInstruct.find("{") >= 0

    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        systemPrompt = super().makeSystemPrompt(args=args, **kwargs)
        if "skills" in kwargs and not self.instruction.syntaxParser.getObjectOutputSyntax():
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
            如果需要使用技能，请使用 "<skillname> 技能参数</skillname>" 来调用技能。
            例如，"<tool1> 参数1, 参数2</tool1>" 
                 "<tool2> 参数3</tool2>"
            '''
        return systemPrompt
        
    def makePrompt(self, useTool: bool = False, **args) -> str:
        result = self.instruction.formattedInstruct
        if self.instruction.syntaxParser.getJsonFormat():
            result += f"""
            请按照下述语义严格以 jsonify 格式输出结果：{self.instruction.syntaxParser.getJsonFormat()}，现在请直接输出 json:
            """
        elif self.instruction.syntaxParser.getJsonListFormat():
            result += f"""
            请按照下述语义严格以 json list 格式输出结果：{self.instruction.syntaxParser.getJsonListFormat()}，现在请直接输出 json list:
            """
        elif self.instruction.syntaxParser.getNormalFormat():
            result += f"""
            请按照下述格式要求输出结果：{self.instruction.syntaxParser.getNormalFormat()}，现在请直接输出结果:
            """
        elif self.instruction.syntaxParser.getObjectOutputSyntax():
            result += f"""
            请必须调用工具，按照工具调用格式输出:
            """
        elif not useTool:
            result += f"""
            请直接输出结果，不要输出任何其他内容:
            """
        return result

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
        self.llmBlock = llmBlock
        self.syntaxParser = SyntaxParser(
            label=label,
            settings=llmBlock.globalContext.settings,
            instruction=curInstruct, 
            repoFuncs=llmBlock.repoFuncs,
            globalObjectTypes=llmBlock.globalObjectTypes)

        self.curInstruct = self.syntaxParser.getInstruction()
        self.formattedInstruct = self.curInstruct
        self.onlyFuncCall = False
        self.isNaiveType = False
        self.noCache = False
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.id = self._createId()

        self.stepInstAnalysis = StepLLMInstrAnalysis(
            instruction=self)
        self.llm :EnhancedLLM = None
        self.reasoning :Reasoning = None
        self.toolkit :Toolkit = None

    def getId(self):
        return self.id

    def execute(
            self, 
            context: Context,
            args :dict, 
            **kwargs) -> InstructResult:
        self.codeInterpreter = CodeInterpreter(context.globalContext)
        self.varDict = self.llmBlock.getVarDict()
        try:
            self._formatCurInstruct(args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"fail parse instruct[{self.curInstruct}]: {str(e)}")

        if self.noCache:
            kwargs["no_cache"] = True

        if self.onlyFuncCall or self.isNaiveType:
            # in this case, the response is the result of the only function call, or a naive type
            return self._processNaiveType(args, **kwargs)

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

        self.stepInstAnalysis.setContext(self.llmBlock.context)
        self.stepInstAnalysis.setLLM(self.getCurLLM())
        self.stepInstAnalysis.setGlobalSkillset(self.llmBlock.globalSkillset)
        if self.syntaxParser.getObjectOutputSyntax():
            self.stepInstAnalysis.setObjectTypes(self.syntaxParser.getObjectOutputSyntax())
            kwargs["tools"] = ObjectTypeFactory.getOpenaiJsonSchema(self.syntaxParser.getObjectOutputSyntax())
        elif self._getToolkit():
            self.stepInstAnalysis.setToolkit(self._getToolkit())
            kwargs["tools"] = self._getToolkit().getTools()

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
            logType="tool",
            toolName=instAnalysisResult.funcExecRecords[0].tool.get_function_name())
        
    def getCurLLM(self) -> EnhancedLLM:
        return self.llm if self.llm else self.syntaxParser.model

    def _getContext(self):
        return self.llmBlock.context

    def _createId(self):
        return f"{self.llmBlock.agentName}_{self.label}_{uuid.uuid4()}"
        
    def __str__(self) -> str:
        return f"Instruction({self.label}): {self.formattedInstruct}"
        
    def _processNaiveType(self, args: dict, **kwargs):
        self._getContext().genResp(self.formattedInstruct, **kwargs)
        kwargs["execNode"].castTo(ExecNodeLLM).addContent(str(self.formattedInstruct))
        return self._createResult(
            response=self.formattedInstruct,
            useTool=False,
            goto=None,
            analysis=self.curInstruct,
            logType="naive")

    def _processGenCode(self, logType: str, args: dict, **kwargs):

        def genCodeAndRun(instruction: Instruction, theArgs: dict):
            result = self.codeInterpreter.execute(
                instruction=instruction.formattedInstruct,
                varDict=instruction.varDict.getAllDict(),
                vm=instruction.llmBlock.context.vm,
                **kwargs)
            instruction.varDict.setLocal(KeywordCurrent, result)
            return Response.buildFrom(result if result else "")

        return self._processWithRetry(genCodeAndRun, args=args, logType=logType)

    def _processPyCode(self, args: dict, **kwargs):
        result = self.llmBlock.context.vm.execPython(
            code=preprocessPyCode(self.formattedInstruct),
            varDict=self.varDict.getAllDict(),
            **kwargs)
        if result == None or not Response.isNaivePyType(result):
            result = ""

        self._getContext().genResp(result, **kwargs)
        kwargs["execNode"].castTo(ExecNodeLLM).addContent(str(result))
        return self._createResult(
            result,
            useTool=False,
            goto=None,
            analysis=self.curInstruct,
            logType="py")
    
    def _processCall(self, args: dict, **kwargs):
        execNodeLLM = kwargs["execNode"].castTo(ExecNodeLLM)
        execNodeSkill :ExecNodeSkill = ExecNodeSkill.build(
            execGraph=execNodeLLM.execGraph,
            execNodeLLM=execNodeLLM,
            skillName=self.syntaxParser.callObj,
            query=self.syntaxParser.callArg,
            skillResult=None,
            label=ExecNodeLabel.AGENT)
        
        def callFunc(instruction: Instruction, theArgs: dict):
            context = instruction.llmBlock.context
            args = {**context.getVarDict().getAllDict(), **theArgs["args"]}
            return context.getEnv().execute(
                agentName=instruction.syntaxParser.callObj,
                context=context,
                query=theArgs["query"].format(**args),
                args=args,
                **{"execNode" : execNodeSkill})

        result = self._processWithRetry(lambdaFunc=callFunc, args=args, logType="call")
        execNodeSkill.setSkillResult(result.response.resp)
        return result

    def _processRet(self, args: dict, **kwargs):
        self._getContext().genResp(self.formattedInstruct, **kwargs)
        kwargs["execNode"].castTo(ExecNodeLLM).addContent(str(self.formattedInstruct))
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
            self._getContext().genResp(f"[trial {i}] execute call {self.syntaxParser.callObj} with query {query} ==> ", **kwargs)
                
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
                vm=self.llmBlock.context.vm,
                retry=True,
                contextLen=self.syntaxParser.model.getContextWindow())
            if not instrOutput.hasError():
                self._getContext().genResp(f"==> {resp.resp}", **kwargs)
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
        logMessage = f"instrExec({self.label}|{self.getCurLLM().model_name}|{logType}): instr[{self.formattedInstruct}] "
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

    def _formatCurInstruct(self, args :dict, **kwargs):
        allArgs = addDict(args, self.varDict.getAllDict())

        #call functions
        curInstruct = self.curInstruct
        if len(self.syntaxParser.funcsToCall) > 0:
            for funcBlock in self.syntaxParser.funcsToCall:
                resp = funcBlock.execute(
                    context=self.llmBlock.context, 
                    query=None, 
                    args=allArgs,
                    **{**kwargs, "curInstruction": self})
                if curInstruct.strip() == funcBlock.getFuncCallPattern().strip():
                    self.onlyFuncCall = True
                    self.formattedInstruct = resp.resp
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

        assert execNodeParent.type == ExecNodeType.SEQUENCE
        execNodeSequence :ExecNodeSequence = execNodeParent
        while curIdx < len(self.instructions):
            curInstruction = self.instructions[curIdx]
            if len(curInstruction.curInstruct.strip()) == 0 and \
                    curInstruction.syntaxParser.flag == SyntaxParser.Flag.RET:
                break

            self.context.genResp(f"\n{curInstruction.label} -> ", **kwargs)

            curInstruction.syntaxParser.reset()

            execNode = ExecNodeLLM.build(
                execGraph=execNodeSequence.execGraph,
                execNodeSequence=execNodeSequence,
                instructionId=curInstruction.getId(),
                curInstruct=curInstruction.curInstruct)

            instructResult = self._step(
                    context=context,
                    instruction=curInstruction, 
                    args=taskArgs, 
                    **{**kwargs, "execNode" : execNode})

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
                vm=self.llmBlock.context.vm,
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
            context: Context,
            instruction: Instruction, 
            args: dict, 
            **kwargs) -> InstructResult:
        t0 = time.time()
        instructResult = instruction.execute(context=context, args=args, **kwargs)
        INFO(logger, instructResult.logMessage + f" costSec[{time.time() - t0:.2f}]")
        return instructResult

class LLMBlock(BaseBlock):

    def __init__(
            self,
            agentName: str,
            globalContext: GlobalContext = None,
            config: str = None,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> None:
        super().__init__(
            agentName=agentName,
            globalContext=globalContext, 
            config=config, 
            toolkit=toolkit, 
            repoFuncs=repoFuncs)

        self.systemPrompt = Loader.load(self.globalContext.settings.llmBasicConfig.systemPrompt)

        if taskExpr:
            self.task = taskExpr    
        else:
            self.task = Loader.load(task) if task else None

        self.jsonKeys = jsonKeys
        self.decomposeTask = decomposeTask
        self.taskEngine = TaskEngine(self, self.task)
        self.instructions = []
        self.stepLLMExtractor = StepLLMExtractor(
            globalContext=self.globalContext)
        self.globalSkillset = self.getEnv().globalSkillset
        self.globalObjectTypes = self.getEnv().globalObjectTypes

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
            query: str,
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
            globalContext: GlobalContext = None,
            config: str = None,
            task: str = None,
            taskExpr: str = None,
            toolkit: Toolkit = None,
            jsonKeys: list = None,
            decomposeTask: bool = True,
            repoFuncs=None) -> 'LLMBlock':
        return LLMBlock(
            agentName=agentName,
            globalContext=globalContext,
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
