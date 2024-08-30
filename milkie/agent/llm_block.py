from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod

import logging
import re
from typing import List

from llama_index.core import Response

from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.functions.base import BaseToolkit, FuncExecRecord
from milkie.functions.sample_tool_kits import SampleToolKit
from milkie.llm.inference import chat
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker

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

class PromptMaker:
    
    def __init__(self) -> None:
        self.task :str = None
        self.taskDetails :str = None
        self.instruction :str = None
        self.instructionDetails :str = None
        self.toolsDesc :str = None
        self.optionDecompose = False
        self.prev = None

    def addTask(self, task :str):
        self.task = task

    def addTaskDetails(self, details: str):
        self.taskDetails = details

    def addInstruction(self, instruction: str):
        self.instruction = instruction

    def addInstructionDetails(self, details: str):
        self.instructionDetails = details

    def addPrev(self, prev):
        self.prev = prev

    def addToolsDesc(self, toolsDesc :str):
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
        你当前的指令是： {self.instruction}
        """

        resultPrompt += "前序指令情况如下:\n"
        resultPrompt += "```\n"
        for i, instructionRecord in enumerate(instructionRecords):
            resultPrompt += f"总结[{i}]: {instructionRecord.synthesizedInstructionRecord}\n"
            resultPrompt += f"详情[{i}]: {instructionRecord.result.response}\n"
        resultPrompt += "```\n"
        return resultPrompt

class Step(ABC):

    def __init__(self, llmBlock :LLMBlock, promptMaker :PromptMaker) -> None:
        self.llmBlock = llmBlock
        self.promptMaker = promptMaker
    
    def run(self, args :dict={}, **kwargs):
        return self.formatResult(self.llmCall(args, **kwargs))
    
    @abstractmethod
    def makePrompt(self, **args) -> str:
        pass

    def llmCall(self, args :dict, **kwargs) -> Response:
        self.prompt = self.makePrompt(**args)
        return chat(
            llm=self.llmBlock.context.globalContext.settings.llm, 
            systemPrompt=self.llmBlock.systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            **kwargs)

    def formatResult(self, result :Response):
        pass

class StepAnalysis(Step):
    def makePrompt(self, **args) -> str:
        resultPrompt = self.promptMaker.promptForTask(**args)
        resultPrompt += f"""
        如果任务目标是逐点表述，请根据任务目标字面表述按照以下格式拆为逐条指令。
        如果任务目标没有逐点表述，仅保留原始任务即可。
        注意：请不要根据自己对任务目标的理解进行拆解!仅仅根据字面的逐点指示进行拆解!

        示例如下
        ``` 
        任务目标：
        请帮我计算下，1 和 2 的平均数是多少
        
        拆解结果：
        1. **指令 1**
        - 计算 1 和 2 的平均数是多少
        
        任务目标：
        请执行任务 1.调用天气查询工具查询北京明天的天气 2.用结果写一篇短文
        
        拆解结果：
        1. **指令 1**
        - 调用天气查询工具查询北京明天的天气
        
        2. **指令 2**
        - 用结果写一篇短文

        ...
        ```
        """
        return resultPrompt

    def formatResult(self, result :Response):
        pattern = re.compile(r'\d+\.\s\*\*(.*?)\*\*\n((\s*-\s+.*\n)*)')
        matches = pattern.findall(result.response)
        instructions = []
        lastInstruction :Instruction = None
        for match in matches:
            instruction = match[1].strip()
            curInstruction = Instruction(
                llmBlock=self.llmBlock, 
                curInstruct=instruction, 
                prev=lastInstruction,
                instructionRecords=self.llmBlock.taskEngine.instructionRecords)
            instructions.append(curInstruction)
            lastInstruction = curInstruction
        return AnalysisResult(
            AnalysisResult.Result.DECOMPOSE,
            instructions=instructions
        )

class StepInstAnalysis(Step):
    def __init__(
            self, 
            llmBlock :LLMBlock, 
            promptMaker :PromptMaker,
            instructionRecords :list[InstructionRecord]) -> None:
        super().__init__(llmBlock, promptMaker)
        self.instructionRecords = instructionRecords
        
    def makePrompt(self, **args) -> str:
        resultPrompt = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)
        resultPrompt += "请给出当前指令的执行结果:"
        return resultPrompt

    def formatResult(self, result :Response) -> InstAnalysisResult:
        chatCompletion = result.metadata["chatCompletion"]
        if chatCompletion.choices[0].message.tool_calls:
            toolCalls = chatCompletion.choices[0].message.tool_calls
            funcExecRecords = self.llmBlock.tools.exec(toolCalls)
            return InstAnalysisResult(
                InstAnalysisResult.Result.TOOL,
                funcExecRecords=funcExecRecords,
                response=chatCompletion.choices[0].message.tool_calls)

        return InstAnalysisResult(
            InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response=result.response)

class StepInstAnswer(Step):
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            promptMaker: PromptMaker,
            instructionRecords :list[InstructionRecord]) -> None:
        super().__init__(llmBlock, promptMaker)
        self.instructionRecords = instructionRecords
        
    def makePrompt(self, **args) -> str:
        resultPrompt = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)

        if funcExecRecords := args.pop("funcExecRecords", None):
            resultPrompt += "函数执行结果如下:\n"
            resultPrompt += "\n".join(f"{funcExecRecord}" for funcExecRecord in funcExecRecords)
        
        resultPrompt += """
        请根据指令中的信息直接对该指令按如下格式给出答案
        ```
        指令回答：XXX
        ```
        """
        return resultPrompt.strip()
    
    def formatResult(self, result :Response):
        if "指令回答" in result.response:
            pattern = re.compile(r'指令回答：(.+)\n?')
            match = pattern.findall(result.response)
            if match:
                return AnswerResult(
                    AnswerResult.Result.ANSWER,
                    response=match[0])
        return AnswerResult(
            AnswerResult.Result.NOANS,
            response=None)

class StepSynthsisInstructionRecord(Step):
    def __init__(
            self, 
            llmBlock :LLMBlock, 
            promptMaker :PromptMaker,
            instructionRecord :InstructionRecord) -> None:
        super().__init__(llmBlock, promptMaker)
        self.instructionRecord = instructionRecord

    def makePrompt(self, **args) -> str:
        resultPrompt = f"""
        指令为：
        {self.instructionRecord.instruction.curInstruct}

        指令执行结果为：
        {self.instructionRecord.result.response}

        请将指令指令本身和执行结果总结为一句话，请直接给出总结结果，总结结果为：
        """
        return resultPrompt

    def formatResult(self, result :Response):
        return result.response

class InstructResult:
    def __init__(
            self, 
            response :Response,
            useTool :bool = False) -> None:
        self.response = response
        self.useTool = useTool

class Instruction:
    def __init__(
            self, 
            llmBlock: LLMBlock, 
            curInstruct: str,
            observation: str = None,
            prev = None,
            instructionRecords :list[InstructionRecord] = None) -> None:
        self.llmBlock = llmBlock
        self.curInstruct = curInstruct
        self.prev: Instruction = prev
        self.observation = observation

        self.promptMaker = PromptMaker()
        self.promptMaker.addTask(llmBlock.taskEngine.task)
        self.promptMaker.addInstruction(self.curInstruct)
        self.promptMaker.addPrev(self.prev)
        self.promptMaker.addToolsDesc(self.llmBlock.tools.getToolsDesc())

        self.stepInstAnalysis = StepInstAnalysis(
            llmBlock=self.llmBlock, 
            promptMaker=self.promptMaker,
            instructionRecords=instructionRecords)
        self.stepInstAnswer = StepInstAnswer(
            llmBlock=self.llmBlock, 
            promptMaker=self.promptMaker,
            instructionRecords=instructionRecords)

    def execute(self, args :dict) -> InstructResult:
        instAnalysisResult = self.stepInstAnalysis.run(
            args=args,
            tools=self.llmBlock.tools.getToolsSchema()
        )

        if instAnalysisResult.result == InstAnalysisResult.Result.ANSWER:
            resp = instAnalysisResult.response.replace('\n', '')
            logger.info(f"instrExec: instr[{self.curInstruct}] ans[{resp}]")
            return self._create_result(
                instAnalysisResult.response, 
                useTool=False,
                analysis=instAnalysisResult.response)
        
        #args["funcExecRecords"] = instAnalysisResult.funcExecRecords
        #instAnswerResult = self.stepInstAnswer.run(args=args)
        #if instAnswerResult.result == AnswerResult.Result.ANSWER:
            #resp = instAnswerResult.response.replace('\n', '')
            #logger.info(f"instrExec: instr[{self.curInstruct}] ans[{resp}] "
                        #f"tools[{instAnalysisResult.funcExecRecords[0].tool.get_function_name()}]")
            #return self._create_result(
                #instAnswerResult.response, 
                #analysis=instAnalysisResult.response, 
                #answer=instAnswerResult.response)

        return self._create_result(
            instAnalysisResult.funcExecRecords[0].result,
            useTool=True,
            analysis=instAnalysisResult.response,
            answer=instAnalysisResult.funcExecRecords[0].result)
        
    def _create_result(self, response: str, useTool :bool, **log_data) -> InstructResult:
        self.llmBlock.context.reqTrace.add("instruction", log_data)
        return InstructResult(Response(response=response), useTool=useTool)

class InstructionRecord:
    def __init__(self, instruction :Instruction, result :InstructResult) -> None:
        self.instruction = instruction
        self.result = result

        if result.useTool:
            self.stepSynthsisInstructionRecord = StepSynthsisInstructionRecord(
                llmBlock=instruction.llmBlock, 
                promptMaker=instruction.promptMaker,
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
        self.lastInstruction :Instruction = None

        self.promptMaker = PromptMaker()
        self.promptMaker.addTask(self.task)

        self.stepAnalysis = StepAnalysis(
            llmBlock=self.llmBlock, 
            promptMaker=self.promptMaker)

        self.instructionRecords :list[InstructionRecord] = []

    def execute(self, taskArgs :dict) -> tuple:
        analysisResult = self.stepAnalysis.run(args=taskArgs)
        if analysisResult.result == AnalysisResult.Result.DECOMPOSE:
            self.instructions = analysisResult.instructions
        else:
            return (False, None)

        while len(self.instructions) != 0:
            instructResult = self._step(args=taskArgs)
            if not instructResult:
                raise ValueError(f"TaskEngine failed to execute")
            
            self.instructionRecords.append(
                InstructionRecord(
                    instruction=self.lastInstruction, 
                    result=instructResult))
        return (True, instructResult.response)

    def _step(self, args :dict) -> InstructResult:
        instruction = self.instructions.pop(0)
        self.lastInstruction = instruction
        return instruction.execute(args=args)

class LLMBlock(BaseBlock):

    def __init__(
            self,
            context :Context = None,
            config :str = None,
            prompt :str = None,
            promptExpr :str = None,
            tools :BaseToolkit = None,
            jsonKeys :list = None) -> None:
        super().__init__(context, config)

        self.systemPrompt = self.context.globalContext.globalConfig.getLLMConfig().systemPrompt

        if promptExpr:
            self.prompt = promptExpr    
        else:
            prompt = prompt if prompt else self.config.prompt
            self.prompt = Loader.load(prompt) if prompt else None

        self.tools = tools
        self.jsonKeys = jsonKeys

        self.taskEngine = TaskEngine(self, self.prompt)

    def execute(self, query :str, args :dict) -> Response:
        taskArgs = {"query_str": query, **args}
        result = self.taskEngine.execute(taskArgs=taskArgs)
        if result[0]:
            return Response(response=result[1])
        else:
            raise ValueError(f"TaskEngine failed to execute")

if __name__ == "__main__":
    agent = LLMBlock(
        promptExpr="执行任务：{query_str}",
        tools=SampleToolKit())
    result = agent.execute("1.搜索爱数信息 2.把结果中的链接都贴出来 3.获取链接对应的网页内容 4.根据上面的内容写一段介绍", args={})
    print(result.response)