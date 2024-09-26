from __future__ import annotations
from enum import Enum

import json
import logging
import re
from typing import List

from milkie.agent.base_block import BaseBlock
from milkie.config.constant import DefaultUsePrevResult, InstFlagCode, InstFlagDecompose, InstFlagGoto, InstFlagIf, InstFlagPy, InstFlagRet, InstFlagThought, KeyNext, KeyRet, KeyVarDictThought, KeyVarDictThtTask, MaxLenLastStepResult, MaxLenThtTask
from milkie.context import Context
from milkie.functions.toolkits.base_toolkits import BaseToolkit, FuncExecRecord
from milkie.functions.toolkits.sample_toolkits import SampleToolKit
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker
from milkie.llm.step_llm import StepLLM
from milkie.log import INFO, DEBUG, ERROR
from milkie.response import Response
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

class InstFlag:
    
    class Flag(Enum):
        NONE = 1
        RET = 2
        CODE = 3
        IF = 4
        GOTO = 5
        PY = 6
        THOUGHT = 7
        DECOMPOSE = 8

    def __init__(self, instruction: str) -> None:
        self.flag = InstFlag.Flag.NONE
        self.label = None
        self.storeVar = None
        self.outputSyntax = None  # 新增: 初始化 outputSyntax
        self.returnVal = False 
        self.instruction = instruction.strip()

        outputSyntaxMatch = re.search(r'(=>\s*(.+?)\s*)->', instruction)
        if outputSyntaxMatch:
            self.outputSyntax = outputSyntaxMatch.group(2).strip()
            self.instruction = self.instruction.replace(outputSyntaxMatch.group(1), "")

        storeVarMatch = re.search(r'(->\s*([a-zA-Z0-9_]+)\s*)$', instruction)
        if storeVarMatch:
            self.storeVar = storeVarMatch.group(2)
            self.instruction = self.instruction.replace(storeVarMatch.group(1), "")
        elif outputSyntaxMatch and "->" in instruction:
            raise Exception("there seems to be a error in storeVar name")

        self.instruction = self.instruction.strip()
        if InstFlagRet in self.instruction:
            self.flag = InstFlag.Flag.RET
            self.returnVal = self.instruction.startswith(InstFlagRet)
            self.instruction = self.instruction.replace(InstFlagRet, "").strip()
        elif InstFlagCode in self.instruction:
            self.flag = InstFlag.Flag.CODE
            self.instruction = self.instruction.replace(InstFlagCode, "")
        elif InstFlagIf in self.instruction:
            self.flag = InstFlag.Flag.IF
        elif InstFlagGoto in self.instruction:
            self.flag = InstFlag.Flag.GOTO
            goto_parts = self.instruction.split(InstFlagGoto)
            self.label = goto_parts[1].split()[0].strip()
            self.instruction = goto_parts[0].strip() + " ".join(goto_parts[1].split()[1:]).strip()
        elif InstFlagPy in self.instruction:
            self.flag = InstFlag.Flag.PY

            def extractCode(text):
                pattern = r"```(.*?)```"
                matches = re.findall(pattern, text, re.DOTALL)
                return matches[0] if matches else None
            
            self.instruction = extractCode(self.instruction)

            for i in range(len(self.instruction)):
                if self.instruction[i] == ' ':
                    continue
                
                if self.instruction[i] == '\n':
                    self.instruction = self.instruction[i+1:]
                    break

            lines = self.instruction.split('\n')
            minSpaces = min(len(line) - len(line.lstrip(' ')) for line in lines if line.strip())
            cleanedLines = [line[minSpaces:] for line in lines]
            self.instruction = '\n'.join(cleanedLines)
        elif InstFlagThought in self.instruction:
            self.flag = InstFlag.Flag.THOUGHT
            self.instruction = self.instruction.replace(InstFlagThought, "")
        elif InstFlagDecompose in self.instruction:
            self.flag = InstFlag.Flag.DECOMPOSE
            self.instruction = self.instruction.replace(InstFlagDecompose, "")

    def getInstruction(self):
        return self.instruction

    def getOutputSyntax(self):
        return re.sub(r'\{{2,}', '{', re.sub(r'\}{2,}', '}', self.outputSyntax))

class PromptMakerInstruction(PromptMaker):

    def __init__(
            self, 
            toolkit :BaseToolkit, 
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
        sampleToolKit = SampleToolKit(llmBlock.context.globalContext)
        sampleEmailDesc = BaseToolkit.getToolsDescWithSingleFunc(sampleToolKit.sendEmail)
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

class StepLLMInstAnalysis(StepLLMBlock):
    def __init__(
            self, 
            promptMaker :PromptMakerInstruction, 
            llmBlock :LLMBlock, 
            instruction: Instruction) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instruction = instruction
        self.instructionRecords = llmBlock.taskEngine.instructionRecords
        
    def makePrompt(self, **args) -> str:
        if args.get("prompt", None) == InstFlagDecompose:
            result = self.promptMaker.promptForDecompose(self.llmBlock)
        elif args.get("prompt", None) == InstFlagThought:
            result = self.promptMaker.promptForThought(self.llmBlock)
        else:
            result = self.promptMaker.promptForInstruction(
                instructionRecords=self.instructionRecords,
                **args)

        if self.instruction.flag.outputSyntax:
            result += f"""
            请按照下述语义严格以 jsonify 格式输出结果：{self.instruction.flag.getOutputSyntax()}，现在请直接输出 json:
            """
        return result

    def formatResult(self, result :Response) -> InstAnalysisResult:
        needToParse = self.instruction.formattedInstruct.find("{") >= 0
        chatCompletion = result.metadata["chatCompletion"]
        if chatCompletion.choices[0].message.tool_calls:
            toolCalls = chatCompletion.choices[0].message.tool_calls
            funcExecRecords = self.llmBlock.toolkit.exec(
                toolCalls, 
                self.llmBlock.getVarDict(),
                needToParse=needToParse)
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=chatCompletion.choices[0].message.tool_calls)
            
        # if function call is not in tools, but it is an answer
        funcExecRecords = self.llmBlock.toolkit.extractToolFromMsg(
            chatCompletion.choices[0].message.content,
            self.llmBlock.getVarDict(),
            needToParse=needToParse)
        if funcExecRecords:
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=chatCompletion.choices[0].message.content)

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result.resp)

class StepLLMSynthsisInstructionRecord(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock, instructionRecord: InstructionRecord) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instructionRecord = instructionRecord

    def makePrompt(self, **args) -> str:
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

class Instruction:
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            curInstruct: str,
            label: str = None,
            observation: str = None,
            prev = None) -> None:
        self.llmBlock = llmBlock
        self.flag = InstFlag(curInstruct)
        self.curInstruct = self.flag.getInstruction()
        self.formattedInstruct = self.curInstruct
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.instructionRecords = llmBlock.taskEngine.instructionRecords
        self.varDict = llmBlock.getVarDict()  # 新增: 访问 TaskEngine 的 varDict

        self.promptMaker = PromptMakerInstruction(
            toolkit=self.llmBlock.toolkit,
            usePrevResult=self.llmBlock.usePrevResult)
        self.promptMaker.setTask(llmBlock.taskEngine.task)
        self.promptMaker.setOrigInstruction(self.curInstruct)
        self.promptMaker.setPrev(self.prev)
        self.stepInstAnalysis = StepLLMInstAnalysis(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock,
            instruction=self)

    def execute(self, args :dict) -> InstructResult:
        try:
            self._formatCurInstruct(args)
        except Exception as e:
            raise SyntaxError(f"fail parse instruct[{self.curInstruct}]")

        useTool = True
        logTypes = []
        if self.flag.flag == InstFlag.Flag.CODE:
            result = self.llmBlock.toolkit.genCodeAndRun(self.formattedInstruct, self.varDict)
            return self._createResult(
                result,
                useTool=True,
                goto=None,
                analysis=self.curInstruct,
                logType="code")
        elif self.flag.flag == InstFlag.Flag.IF:
            result = self.llmBlock.toolkit.genCodeAndRun(self.formattedInstruct, self.varDict)
            return self._createResult(
                result,
                useTool=True,
                goto=result,
                analysis=self.curInstruct,
                logType="if")
        elif self.flag.flag == InstFlag.Flag.PY:
            def preprocessPyInstruct(instruct: str):
                instruct = instruct.replace("$varDict", "self.varDict")
                instruct = instruct.replace(KeyNext, f'"{KeyNext}"')
                instruct = instruct.replace(KeyRet, f'"{KeyRet}"')
                return instruct

            result = self.llmBlock.toolkit.runCode(
                preprocessPyInstruct(self.formattedInstruct),
                self.varDict)
            return self._createResult(
                result,
                useTool=False,
                goto=None,
                analysis=self.curInstruct,
                logType="py")
        elif self.flag.flag == InstFlag.Flag.RET and self.flag.returnVal:
            return self._createResult(
                self.formattedInstruct,
                useTool=False,
                goto=None,
                analysis=self.curInstruct,
                logType="ret")

        if self.flag.flag == InstFlag.Flag.THOUGHT:
            self.promptMaker.promptForThought(self.llmBlock)
            args["prompt"] = InstFlagThought
            useTool = False
            logTypes.append("thought")
        elif self.flag.flag == InstFlag.Flag.DECOMPOSE:
            self.promptMaker.promptForDecompose(self.llmBlock)
            args["prompt"] = InstFlagDecompose
            useTool = False
            logTypes.append("decompose")
        else:
            args["prompt"] = None

        if useTool:
            instAnalysisResult = self.stepInstAnalysis.run(
                args=args,
                tools=self.llmBlock.toolkit.getToolsSchema()
            )
        else:
            instAnalysisResult = self.stepInstAnalysis.run(
                args=args)

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
        if self.flag.flag == InstFlag.Flag.RET and self.flag.returnVal:
            if response.startswith("{"):
                retJson = json.loads(response)
                retJson = restoreVariablesInDict(retJson, self.varDict)
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

    def _formatCurInstruct(self, args :dict):
        allArgs = {**args, **self.varDict}
        self.formattedInstruct = restoreVariablesInStr(
            self.curInstruct, 
            allArgs)
        if self.flag.flag == InstFlag.Flag.GOTO:
            self.formattedInstruct = self.formattedInstruct.replace(f"#GOTO {self.flag.label}", "")
        self.promptMaker.setFormattedInstruction(self.formattedInstruct)

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

    def execute(self, taskArgs :dict, instructions: list[tuple[str, Instruction]]) -> tuple:
        self.lastInstruction = None
        self.instructions = instructions

        curIdx = 0
        instructResult :InstructResult = None  # 初始化 instructResult
        while curIdx < len(self.instructions):
            label, instruction = self.instructions[curIdx]
            if len(instruction.curInstruct.strip()) == 0 and \
                    instruction.flag.flag == InstFlag.Flag.RET:
                break
            
            instructResult = self._step(instruction, args=taskArgs)
            if instructResult.isRet():
                logger.info("end of the task")
                break

            #set variables
            self.llmBlock.setResp(label, instructResult.response.resp)
            print(f"{label} -> {instructResult.response.resp}")
            if instruction.flag.flag == InstFlag.Flag.THOUGHT:
                self.llmBlock.setVarDict(KeyVarDictThtTask, instruction.formattedInstruct[:MaxLenThtTask])
                self.llmBlock.setVarDict(KeyVarDictThought, instructResult.response.resp)
            
            if instruction.flag.storeVar:
                if instruction.flag.outputSyntax:
                    jsonStr = instructResult.response.respStr.replace("```json", '').replace("```", '').replace("\n", '')
                    try:
                        self.llmBlock.setVarDict(instruction.flag.storeVar, json.loads(jsonStr))
                    except Exception as e:
                        logger.error(f"Error parsing JSON: {e}")
                        self.llmBlock.setVarDict(instruction.flag.storeVar, jsonStr)
                else:
                    self.llmBlock.setVarDict(instruction.flag.storeVar, instructResult.response.resp)

            #append instruction record
            self.instructionRecords.append(
                InstructionRecord(
                    instruction=instruction, 
                    result=instructResult))

            if instruction.flag.flag == InstFlag.Flag.RET:
                break

            # adjust instructions and curIdx
            if instruction.flag.flag == InstFlag.Flag.DECOMPOSE:
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
            elif instruction.flag.flag == InstFlag.Flag.GOTO:
                curIdx = self.getInstrLabels().index(instruction.flag.label)
            else:
                curIdx += 1

        return (True, instructResult.response)

    def getInstrLabels(self) -> list[str]:
        return [label for label, _ in self.instructions]

    def _step(self, instruction: Instruction, args: dict) -> InstructResult:
        instructResult = instruction.execute(args=args)
        return instructResult

class LLMBlock(BaseBlock):

    def __init__(
            self,
            context :Context = None,
            config :str = None,
            usePrevResult :bool = DefaultUsePrevResult,
            task :str = None,
            taskExpr :str = None,
            toolkit :BaseToolkit = None,
            jsonKeys :list = None,
            decomposeTask: bool = True) -> None:
        super().__init__(context, config, toolkit, usePrevResult)

        self.usePrevResult = usePrevResult  # 确保这个属性被设置

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
            query :str=None, 
            args :dict={},
            prevBlock :LLMBlock=None) -> Response:
        self.updateFromPrevBlock(prevBlock, args)

        INFO(logger, f"LLMBlock execute: query[{query}] args[{args}]")

        taskArgs = {"query_str": query, **args}
        if not self.isCompiled:
            self.compile()  # 只在未编译时进行编译
        _, result = self.taskEngine.execute(
            taskArgs=taskArgs, 
            instructions=self.instructions)
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

if __name__ == "__main__":
    task = """
    Dog. #CODE 生成两个随机数 => [第个随机数，第二个随机数] -> random_nums
    Cat. #IF 如果{random_nums.0}是奇数，返回Tiger，如果是偶数，返回Monkey
    Tiger、 根据{random_nums.0}讲个笑话, #GOTO Slime -> joke
    Monkey、 说一首诗 -> poetry
    Fish. #RET {poetry}
    Slime. #RET {{"result": "{joke}"}}
    """
    llmBlock = LLMBlock(taskExpr=task)
    print(llmBlock.execute().resp)
