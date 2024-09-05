from __future__ import annotations
from enum import Enum

import json
import logging
import re
from typing import List

from llama_index.core import Response

from milkie.agent.base_block import BaseBlock
from milkie.config.constant import MaxLenLastStepResult
from milkie.context import Context
from milkie.functions.base import BaseToolkit, FuncExecRecord
from milkie.functions.sample_tool_kits import SampleToolKit
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, DEBUG, ERROR

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

class AnalysisResult:
    class Result(Enum):
        DECOMPOSE = 2
        CANNOT = 3
        ERROR = 4

    def __init__(
            self,
            result: Result,
            instructions: list=None):
        self.result = result
        self.instructions = instructions

    def instrSummary(self):
        result = ""
        for _, instruction in self.instructions:
            result += f"{instruction.label}: {instruction.curInstruct}\n"
        return result

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

class InstFlag:
    
    class Flag(Enum):
        NONE = 1
        END = 2
        CODE = 3
        IF = 4
        GOTO = 5
        PY = 6

    def __init__(self, instruction: str) -> None:
        self.flag = InstFlag.Flag.NONE
        self.label = None
        self.storeVar = None
        self.outputSyntax = None  # 新增: 初始化 outputSyntax
        self.instruction = instruction

        if "#END" in instruction:
            self.flag = InstFlag.Flag.END
            self.instruction = instruction.replace("#END", "")
        elif "#CODE" in instruction:
            self.flag = InstFlag.Flag.CODE
            self.instruction = instruction.replace("#CODE", "")
        elif "#IF" in instruction:
            self.flag = InstFlag.Flag.IF
        elif "#GOTO" in instruction:
            self.flag = InstFlag.Flag.GOTO
            self.label = instruction.split("GOTO")[1].split()[0].strip()
            self.instruction = instruction.replace("#GOTO", "")
        elif "#PY" in instruction:
            self.flag = InstFlag.Flag.PY
            self.instruction = instruction.replace("#PY", "").strip()
            if not self.instruction.startswith("```") and not self.instruction.endswith("```"):
                raise Exception("python code must be wrapped by ```")
            self.instruction = self.instruction[3:-3]

        # 新增: 解析 outputSyntax
        outputSyntaxMatch = re.search(r'=>\s*(.+?)\s*->', instruction)
        if outputSyntaxMatch:
            self.outputSyntax = outputSyntaxMatch.group(1).strip()

        storeVarMatch = re.search(r'->\s*([a-zA-Z0-9_]+)$', instruction)
        if storeVarMatch:
            self.storeVar = storeVarMatch.group(1)
        elif outputSyntaxMatch and "->" in instruction:
            raise Exception("there seems to be a error in storeVar name")

    def getInstruction(self):
        return self.instruction

    def getOutputSyntax(self):
        return re.sub(r'\{{2,}', '{', re.sub(r'\}{2,}', '}', self.outputSyntax))


class PromptMaker:

    def __init__(self) -> None:
        self.task :str = None
        self.taskDetails :str = None
        self.origInstruction :str = None
        self.formattedInstruction :str = None
        self.instructionDetails :str = None
        self.toolsDesc :str = None
        self.optionDecompose = False
        self.prev = None

    def setTask(self, task: str):
        self.task = task

    def setTaskDetails(self, details: str):
        self.taskDetails = details

    def setOrigInstruction(self, instruction: str):
        self.origInstruction = instruction
        self.formattedInstruction = instruction

    def setFormattedInstruction(self, instruction: str):
        self.formattedInstruction = instruction

    def setInstructionDetails(self, details: str):
        self.instructionDetails = details

    def setPrev(self, prev):
        self.prev = prev

    def setToolsDesc(self, toolsDesc: str):
        self.toolsDesc = toolsDesc

    def promptForTask(self, **args):
        resultPrompt = f"""
        你的任务目标是： {self.task.format(**args)}
        """

        if self.taskDetails:
            resultPrompt += f"""
            任务详情：
            {self.taskDetails.format(**args)}
            """
        return resultPrompt

    def promptForInstruction(
            self, 
            instructionRecords :list[InstructionRecord],
            **args):
        resultPrompt = f"""
        你当前的指令是： {self.origInstruction}
        """

        resultPrompt += "前序指令情况如下:\n"
        resultPrompt += "```\n"
        resultPrompt += self.prevStepSummary(instructionRecords)
        resultPrompt += "```\n"

        resultPrompt += f"""
        你当前的指令是： {self.formattedInstruction}
        """
        return resultPrompt

    def prevStepSummary(self, instructionRecords :list[InstructionRecord]):
        result = ""
        if len(instructionRecords) > 0:
            result += f"上一步总结: {instructionRecords[-1].synthesizedInstructionRecord[:MaxLenLastStepResult]}\n"
            result += f"上一步详情: {instructionRecords[-1].result.response.response[:MaxLenLastStepResult]}\n"
        return result

class StepLLMBlock(StepLLM):
    def __init__(self, promptMaker, llmBlock :LLMBlock):
        super().__init__(llmBlock.context.globalContext, promptMaker)
        self.llmBlock = llmBlock

class StepLLMInstAnalysis(StepLLMBlock):
    def __init__(
            self, 
            promptMaker, 
            llmBlock, 
            instruction: Instruction,
            instructionRecords: list[InstructionRecord]) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instruction = instruction
        self.instructionRecords = instructionRecords
        
    def makePrompt(self, **args) -> str:
        result = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)
        if self.instruction.flag.outputSyntax:
            result += f"""
            请以 json 格式输出结果，输出语法为：{self.instruction.flag.getOutputSyntax()}，现在请直接输出 json:
            """
        return result

    def formatResult(self, result :Response) -> InstAnalysisResult:
        chatCompletion = result.metadata["chatCompletion"]
        if chatCompletion.choices[0].message.tool_calls:
            toolCalls = chatCompletion.choices[0].message.tool_calls
            funcExecRecords = self.llmBlock.tools.exec(toolCalls, self.llmBlock.getVarDict())
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=chatCompletion.choices[0].message.tool_calls)

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result.response)

class StepLLMSynthsisInstructionRecord(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock, instructionRecord: InstructionRecord) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instructionRecord = instructionRecord

    def makePrompt(self, **args) -> str:
        resultPrompt = f"""
        指令为：
        {self.instructionRecord.instruction.curInstruct}

        指令执行结果为：
        {self.instructionRecord.result.response.response}

        请将指令指令本身和执行结果总结为一句话，请直接给出总结结果，总结结果为：
        """
        return resultPrompt

    def formatResult(self, result :Response):
        return result.response

class InstructResult:
    def __init__(
            self, 
            response :Response,
            goto :str = None,
            useTool :bool = False) -> None:
        self.response = response
        self.goto = goto
        self.useTool = useTool

    def isEnd(self):
        return self.response.response == "_END_"

    def isNext(self):
        return self.response.response == "_NEXT_"

class Instruction:
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            curInstruct: str,
            label: str = None,
            observation: str = None,
            prev = None,
            instructionRecords :list[InstructionRecord] = None) -> None:
        self.llmBlock = llmBlock
        self.flag = InstFlag(curInstruct)
        self.curInstruct = self.flag.getInstruction()
        self.formattedInstruct = self.curInstruct
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.instructionRecords = instructionRecords
        self.varDict = llmBlock.taskEngine.varDict  # 新增: 访问 TaskEngine 的 varDict

        self.promptMaker = PromptMaker()
        self.promptMaker.setTask(llmBlock.taskEngine.task)
        self.promptMaker.setOrigInstruction(self.curInstruct)
        self.promptMaker.setPrev(self.prev)
        self.promptMaker.setToolsDesc(self.llmBlock.tools.getToolsDesc())
        self.stepInstAnalysis = StepLLMInstAnalysis(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock,
            instruction=self,
            instructionRecords=instructionRecords)

    def execute(self, args :dict) -> InstructResult:
        self._formatCurInstruct()
        if self.flag.flag == InstFlag.Flag.CODE:
            result = self.llmBlock.tools.genCodeAndRun(self.formattedInstruct)
            return self._createResult(
                result,
                useTool=True,
                goto=None,
                analysis=self.curInstruct,
                logType="code")
        elif self.flag.flag == InstFlag.Flag.IF:
            result = self.llmBlock.tools.genCodeAndRun(self.formattedInstruct)
            return self._createResult(
                result,
                useTool=True,
                goto=result,
                analysis=self.curInstruct,
                logType="if")
        elif self.flag.flag == InstFlag.Flag.PY:
            def preprocessPyInstruct(instruct: str):
                instruct = instruct.replace("$varDict", "self.varDict")
                instruct = instruct.replace("_NEXT_", '"_NEXT_"')
                instruct = instruct.replace("_END_", '"_END_"')
                return instruct

            result = self.llmBlock.tools.runCode(
                preprocessPyInstruct(self.formattedInstruct))
            return self._createResult(
                result,
                useTool=False,
                goto=result,
                analysis=self.curInstruct,
                logType="py")
        
        instAnalysisResult = self.stepInstAnalysis.run(
            args=args,
            tools=self.llmBlock.tools.getToolsSchema()
        )

        if instAnalysisResult.result == InstAnalysisResult.Result.ANSWER:
            return self._createResult(
                instAnalysisResult.response, 
                useTool=False,
                goto=None,
                analysis=instAnalysisResult.response,
                logType="ans")

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
            response: str, 
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
        return InstructResult(
            Response(response=response), 
            useTool=useTool,
            goto=goto)

    def _formatCurInstruct(self):
        self.formattedInstruct = self._nestedFormat(self.curInstruct, self.varDict)
        if self.flag.flag == InstFlag.Flag.GOTO:
            self.formattedInstruct = self.formattedInstruct.replace(f"#GOTO {self.flag.label}", "")
        self.promptMaker.setFormattedInstruction(self.formattedInstruct)

    @staticmethod
    def _nestedFormat(template, data):
        from string import Formatter

        def recursive_lookup(d, key):
            keys = key.split('.')
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k)
                elif isinstance(d, list):
                    try:
                        d = d[int(k)]
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
            return d

        class NestedFormatter(Formatter):
            def get_field(self, field_name, args, kwargs):
                return recursive_lookup(data, field_name), field_name

        return NestedFormatter().format(template)

class InstructionRecord:
    def __init__(self, instruction :Instruction, result :InstructResult) -> None:
        self.instruction = instruction
        self.result = result

        if result.useTool:
            self.stepSynthsisInstructionRecord = StepLLMSynthsisInstructionRecord(
                promptMaker=instruction.promptMaker,
                llmBlock=instruction.llmBlock,
                instructionRecord=self)
            self.synthesizedInstructionRecord = self.stepSynthsisInstructionRecord.run()
        else:
            self.synthesizedInstructionRecord = self.result.response.response

class TaskEngine:
    def __init__(
            self, 
            llmBlock :LLMBlock,
            task :str) -> None:
        self.llmBlock = llmBlock
        self.task = task
        self.instructions :list[tuple[str, Instruction]] = []
        self.lastInstruction :Instruction = None

        self.promptMaker = PromptMaker()
        self.promptMaker.setTask(self.task)

        self.instructionRecords :list[InstructionRecord] = []
        self.varDict = {"_resp": {}}  # 修改: 初始化 varDict 和 _resp 子字典

    def execute(self, taskArgs :dict, decomposeTask :bool=True) -> tuple:
        self.instructionRecords.clear()
        self.lastInstruction = None
        self.varDict["_resp"] = {}

        if decomposeTask:
            analysisResult = self._decomposeTask(self.task.format(**taskArgs))
        else:
            analysisResult = AnalysisResult(
                AnalysisResult.Result.DECOMPOSE, 
                instructions=[(1, self.task.format(**taskArgs).strip().replace('\n', '//'))])

        if analysisResult.result == AnalysisResult.Result.DECOMPOSE:
            for i in range(len(analysisResult.instructions)):
                label, instruction = analysisResult.instructions[i]
                analysisResult.instructions[i] = (
                    label, 
                    Instruction(
                        llmBlock=self.llmBlock, 
                        curInstruct=instruction, 
                        label=label,
                        prev=self.lastInstruction,
                        instructionRecords=self.instructionRecords))
            self.instructions = analysisResult.instructions
            INFO(logger, f"analysis ans[{analysisResult.instrSummary()}]")
        else:
            return (False, None)

        self.curIdx = 0
        while self.curIdx < len(self.instructions):
            label, instruction = self.instructions[self.curIdx]

            if len(instruction.curInstruct.strip()) == 0 and \
                    instruction.flag.flag == InstFlag.Flag.END:
                break
            
            instructResult = self._step(args=taskArgs)
            if instructResult.isEnd():
                logger.info("end of the task")
                break
            
            #set varDict
            self.varDict["_resp"][label] = instructResult.response.response
            if self.lastInstruction.flag.storeVar:
                if self.lastInstruction.flag.outputSyntax:
                    jsonStr = instructResult.response.response.replace("```json", '').replace("```", '').replace("\n", '')
                    try:
                        self.varDict[self.lastInstruction.flag.storeVar] = json.loads(jsonStr)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                else:
                    self.varDict[self.lastInstruction.flag.storeVar] = instructResult.response.response

            #add instruction record
            self.instructionRecords.append(
                InstructionRecord(
                    instruction=self.lastInstruction, 
                    result=instructResult))

            if self.lastInstruction.flag.flag == InstFlag.Flag.END:
                break

        return (True, instructResult.response)

    def getInstrLabels(self) -> list[str]:
        return [label for label, _ in self.instructions]

    def _step(self, args :dict) -> InstructResult:
        _, instruction = self.instructions[self.curIdx]
        self.lastInstruction = instruction
        instructResult = instruction.execute(args=args)

        if instructResult.isEnd():
            return instructResult
        elif instructResult.isNext():
            self.curIdx += 1
            return instructResult

        #process goto
        if instructResult.goto:
            self.curIdx = self.getInstrLabels().index(instructResult.goto)
        elif instruction.flag.flag == InstFlag.Flag.GOTO:
            self.curIdx = self.getInstrLabels().index(instruction.flag.label)
        else:
            self.curIdx += 1
        return instructResult

    @staticmethod
    def _decomposeTask(task: str) -> AnalysisResult:
        label_pattern = r'^([\w\u4e00-\u9fff]+)[.、]\s+'
        
        lines = task.split('\n')
        instructions = []
        current_label = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(label_pattern, line)
            if match:
                # 如果有之前的内容，保存它
                if current_label is not None:
                    instructions.append((current_label, '//'.join(current_content)))
                
                # 开始新的指令
                current_label = match.group(1)
                current_content = [line[len(match.group(0)):].strip()]
            else:
                # 继续添加到当前内容
                current_content.append(line)

        # 添加最后一个指令
        if current_label is not None:
            instructions.append((current_label, '//'.join(current_content)))

        # 如果没有找到任何指令，将整个任务作为一个指令
        if not instructions:
            instructions.append(("1", task.strip().replace('\n', '//')))

        return AnalysisResult(AnalysisResult.Result.DECOMPOSE, instructions=instructions)

class LLMBlock(BaseBlock):

    def __init__(
            self,
            context :Context = None,
            config :str = None,
            task :str = None,
            taskExpr :str = None,
            tools :BaseToolkit = None,
            jsonKeys :list = None) -> None:
        super().__init__(context, config)

        self.systemPrompt = self.context.globalContext.globalConfig.getLLMConfig().systemPrompt

        if taskExpr:
            self.task = taskExpr    
        else:
            task = task if task else self.config.task
            self.task = Loader.load(task) if task else None

        self.tools :BaseToolkit = tools if tools else SampleToolKit(self.context.globalContext)
        self.jsonKeys = jsonKeys

        self.taskEngine = TaskEngine(self, self.task)

    def execute(
            self, 
            query :str, 
            args :dict={}, 
            decomposeTask :bool=True) -> Response:
        INFO(logger, f"LLMBlock execute: query[{query}] args[{args}]")
        
        taskArgs = {"query_str": query, **args}
        try:
            result = self.taskEngine.execute(taskArgs=taskArgs, decomposeTask=decomposeTask)
            if result[0]:
                return Response(
                    response=result[1], 
                    metadata=self.taskEngine.varDict)
        except Exception as e:
            raise ValueError(f"TaskEngine failed to execute, reason: {e}")
    
    def getVarDict(self):
        return self.taskEngine.varDict
    
    def setVarDict(self, key :str, val):
        self.taskEngine.varDict[key] = val
    
    def clearVarDict(self):
        self.taskEngine.varDict.clear()

if __name__ == "__main__":
    llmBlock = LLMBlock(taskExpr="执行任务：{query_str}")

    print(llmBlock.execute("""
        Dog. #CODE 生成两个随机数 => [第个随机数，第二个随机数] -> random_nums
        Cat. #IF 如果{random_nums.0}是奇数，返回Tiger，如果是偶数，返回Monkey
        Tiger、 根据{random_nums.0}讲个笑话, #GOTO Slime
        Monkey、 说一首诗
        Slime. 把上一步结果中的创作内容输出
        """).response)
#        
#    print(agent.execute(
#        "1.搜索爱数信息 2.把结果中的链接都贴出来 3.获取链接对应的网页内容 4.根据上面的内容一段介绍"
#        ).response)
