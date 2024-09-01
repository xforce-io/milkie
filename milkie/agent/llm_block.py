from __future__ import annotations
from enum import Enum

import logging
import re
from typing import List

from llama_index.core import Response

from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.functions.base import BaseToolkit, FuncExecRecord
from milkie.functions.sample_tool_kits import SampleToolKit
from milkie.prompt.prompt import Loader
from milkie.prompt.prompt_maker import PromptMaker
from milkie.llm.step_llm import StepLLM

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

    def __init__(self, instruction :str) -> None:
        self.flag = InstFlag.Flag.NONE
        self.label = None

        if "#END" in instruction:
            self.flag = InstFlag.Flag.END
        elif "#CODE" in instruction:
            self.flag = InstFlag.Flag.CODE
        elif "IF" in instruction:
            self.flag = InstFlag.Flag.IF
        elif "GOTO" in instruction:
            self.flag = InstFlag.Flag.GOTO
            self.label = instruction.split("GOTO")[1].split()[0].strip()

class PromptMaker:

    MaxLenLastStepResult = 5000
    
    def __init__(self) -> None:
        self.task :str = None
        self.taskDetails :str = None
        self.instruction :str = None
        self.instructionDetails :str = None
        self.toolsDesc :str = None
        self.optionDecompose = False
        self.prev = None

    def setTask(self, task: str):
        self.task = task

    def setTaskDetails(self, details: str):
        self.taskDetails = details

    def setInstruction(self, instruction: str):
        self.instruction = instruction

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
        你当前的指令是： {self.instruction}
        """

        resultPrompt += "前序指令情况如下:\n"
        resultPrompt += "```\n"
        resultPrompt += self.prevStepSummary(instructionRecords)
        resultPrompt += "```\n"

        resultPrompt += f"""
        你当前的指令是： {self.instruction}
        """
        return resultPrompt

    def prevStepSummary(self, instructionRecords :list[InstructionRecord]):
        result = ""
        if len(instructionRecords) > 0:
            result += f"上一步总结: {instructionRecords[0].synthesizedInstructionRecord[:self.MaxLenLastStepResult]}\n"
            result += f"上一步详情: {instructionRecords[0].result.response.response[:self.MaxLenLastStepResult]}\n"
        return result

class StepLLMBlock(StepLLM):
    def __init__(self, promptMaker, llmBlock :LLMBlock):
        super().__init__(llmBlock.context.globalContext, promptMaker)
        self.llmBlock = llmBlock

class StepLLMAnalysis(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock):
        super().__init__(promptMaker, llmBlock)

    def makePrompt(self, **args) -> str:
        resultPrompt = self.promptMaker.promptForTask(**args)
        resultPrompt += f"""
        如果任务目标是逐点表述，请根据任务目标字面表述��照以下格式拆为逐条指令。
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
        请执行任务 
        Bob. 生成个随机数 
        Alice, #IF 如果上一步结果是奇数，跳到 三，如果是偶数，跳到 4 
        Cath. 讲个笑话 #END
        Dave. #CODE用结果写一篇短文
        
        拆解结果：
        1. **指令 Bob**
        - 生成个随机数
          
        2. **指令 Alice**
        - #IF 如果上一步结果是奇数，跳到 三，如果是偶数，跳到 4 
       
        3. **指令 Cath**
        - 讲个笑话 #END
       
        4. **指令 Dave**
        - #CODE用结果写一篇短文

        任务目标：
        请执行任务 
        1. 生成个随机数 
        二、 #IF 如果上一步结果是奇数，跳到 三，如果是偶数，跳到 4 
        三. 讲个笑话 #END
        四、 #CODE用结果写一篇短文
        
        拆解结果：
        1. **指令 1**
        - 生成个随机数
          
        2. **指令 二**
        - #IF 如果上一步结果是奇数，跳到 三，如果是偶数，跳到 4 
       
        3. **指令 三**
        - 讲个笑话 #END
       
        4. **指令 四**
        - #CODE用结果写一篇短文

        ...
        ```
        """
        return resultPrompt

    def formatResult(self, result :Response):
        pattern = re.compile(r'\d+\.\s\*\*指令\s(.*?)\*\*\n((\s*-\s+.*\n)*)')
        matches = pattern.findall(result.response)
        instructions = []
        lastInstruction :Instruction = None
        for match in matches:
            label = match[0].strip()
            instruction = match[1].strip()
            curInstruction = Instruction(
                llmBlock=self.llmBlock, 
                curInstruct=instruction, 
                label=label,
                prev=lastInstruction,
                instructionRecords=self.llmBlock.taskEngine.instructionRecords)
            instructions.append((label, curInstruction))
            lastInstruction = curInstruction
        return AnalysisResult(
            AnalysisResult.Result.DECOMPOSE,
            instructions=instructions
        )

class StepLLMInstAnalysis(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock, instructionRecords: list[InstructionRecord]) -> None:
        super().__init__(promptMaker, llmBlock)
        self.instructionRecords = instructionRecords
        
    def makePrompt(self, **args) -> str:
        resultPrompt = self.promptMaker.promptForInstruction(
            instructionRecords=self.instructionRecords,
            **args)
        resultPrompt += "请直接给出当前指令的执行结果:"
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

class StepLLMInstAnswer(StepLLMBlock):
    def __init__(self, promptMaker, llmBlock, instructionRecords: list[InstructionRecord]) -> None:
        super().__init__(promptMaker, llmBlock)
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
        self.curInstruct = curInstruct
        self.formattedInstruct = curInstruct
        self.label = label
        self.prev: Instruction = prev
        self.observation = observation
        self.instructionRecords = instructionRecords
        self.flag = InstFlag(self.curInstruct)
        self.varDict = llmBlock.taskEngine.varDict  # 新增: 访问 TaskEngine 的 varDict

        self.promptMaker = PromptMaker()
        self.promptMaker.setTask(llmBlock.taskEngine.task)
        self.promptMaker.setInstruction(self.curInstruct)
        self.promptMaker.setPrev(self.prev)
        self.promptMaker.setToolsDesc(self.llmBlock.tools.getToolsDesc())
        self.stepInstAnalysis = StepLLMInstAnalysis(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock,
            instructionRecords=instructionRecords)
        self.stepInstAnswer = StepLLMInstAnswer(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock,
            instructionRecords=instructionRecords)

    def execute(self, args :dict) -> InstructResult:
        self._formatCurInstruct()
        if self.flag.flag == InstFlag.Flag.CODE:
            result = self.llmBlock.tools.runCodeInterpreter(self.formattedInstruct)
            logger.info(f"instrExec({self.label}|code): instr[{self.formattedInstruct}] ans[{result}]")
            return self._create_result(
                result,
                useTool=True,
                goto=None,
                analysis=self.curInstruct)
        elif self.flag.flag == InstFlag.Flag.IF:
            result = self.llmBlock.tools.runCodeInterpreter(self.formattedInstruct)
            instructResult = self._create_result(
                result,
                useTool=True,
                goto=result,
                analysis=self.curInstruct)
            instructResult.goto = result
            logger.info(f"instrExec({self.label}|if): instr[{self.formattedInstruct}] ans[{result}]")
            return instructResult
        
        instAnalysisResult = self.stepInstAnalysis.run(
            args=args,
            tools=self.llmBlock.tools.getToolsSchema()
        )

        if instAnalysisResult.result == InstAnalysisResult.Result.ANSWER:
            resp = instAnalysisResult.response.replace('\n', '')
            logger.info(f"instrExec({self.label}|ans): instr[{self.formattedInstruct}] ans[{resp}]")
            return self._create_result(
                instAnalysisResult.response, 
                useTool=False,
                goto=None,
                analysis=instAnalysisResult.response)

        logger.info(f"instrExec({self.label}|tool): instr[{self.formattedInstruct}] "
                    f"tool[{instAnalysisResult.funcExecRecords[0].tool.get_function_name()}]")
        return self._create_result(
            instAnalysisResult.funcExecRecords[0].result,
            useTool=True,
            goto=None,
            analysis=instAnalysisResult.response,
            answer=instAnalysisResult.funcExecRecords[0].result)
        
    def _create_result(
            self, 
            response: str, 
            useTool :bool, 
            goto :str,
            **log_data) -> InstructResult:
        self.llmBlock.context.reqTrace.add("instruction", log_data)
        return InstructResult(
            Response(response=response), 
            useTool=useTool,
            goto=goto)

    def _formatCurInstruct(self):
        self.formattedInstruct = self._nested_format(self.curInstruct, self.varDict)
        if self.flag.flag == InstFlag.Flag.GOTO:
            self.formattedInstruct = self.formattedInstruct.replace(f"#GOTO {self.flag.label}", "")
        self.promptMaker.setInstruction(self.formattedInstruct)

    @staticmethod
    def _nested_format(template, data):
        from string import Formatter

        def recursive_lookup(d, key):
            keys = key.split('.')
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k)
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

        self.stepAnalysis = StepLLMAnalysis(
            promptMaker=self.promptMaker,
            llmBlock=llmBlock)

        self.instructionRecords :list[InstructionRecord] = []
        self.varDict = {"resp": {}}  # 修改: 初始化 varDict 和 resp 子字典

    def execute(self, taskArgs :dict) -> tuple:
        self.instructionRecords.clear()
        self.varDict["resp"].clear()  # 新增: 清空之前的结果
        analysisResult = self.stepAnalysis.run(args=taskArgs)
        if analysisResult.result == AnalysisResult.Result.DECOMPOSE:
            self.instructions = analysisResult.instructions
            logger.info(f"analysis ans[{analysisResult.instrSummary()}]")
        else:
            return (False, None)

        self.curIdx = 0
        while self.curIdx < len(self.instructions):
            label, instruction = self.instructions[self.curIdx]
            instructResult = self._step(args=taskArgs)
            if not instructResult:
                raise ValueError(f"TaskEngine failed to execute")
            
            # 新增: 存储结果
            self.varDict["resp"][label] = instructResult.response.response

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

        #process goto
        if instructResult.goto:
            self.curIdx = self.getInstrLabels().index(instructResult.goto)
        elif instruction.flag.flag == InstFlag.Flag.GOTO:
            self.curIdx = self.getInstrLabels().index(instruction.flag.label)
        else:
            self.curIdx += 1
        return instructResult

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

        self.tools :BaseToolkit = tools if tools else SampleToolKit(self.context.globalContext)
        self.jsonKeys = jsonKeys

        self.taskEngine = TaskEngine(self, self.prompt)

    def execute(self, query :str, args :dict={}) -> Response:
        logger.info(f"LLMBlock execute: query[{query}] args[{args}]")
        
        taskArgs = {"query_str": query, **args}
        result = self.taskEngine.execute(taskArgs=taskArgs)
        if result[0]:
            return Response(response=result[1], metadata={"resp": self.taskEngine.varDict["resp"]})
        else:
            raise ValueError(f"TaskEngine failed to execute")

if __name__ == "__main__":
    llmBlock = LLMBlock(promptExpr="执行任务：{query_str}")

#    print(agent.execute("""
#        Dog. #CODE 生成一个随机数
#        Cat. #IF 如果{resp.Dog}是奇数，返回Tiger，如果是偶数，返回Monkey
#        Tiger、 根据{resp.Dog}讲个笑话, #GOTO Slime
#        Monkey、 说一首诗
#        Slime. 把上一步结果中的创作内容发送到我的邮箱里,邮箱是 Freeman.xu@aishu.cn, 标题为哈哈哈
#        """).response)
#        
#    print(agent.execute(
#        "1.搜索爱数信息 2.把结果中的链接都贴出来 3.获取链接对应的网页内容 4.根据上面的内容写一段介绍"
#        ).response)
#
#    print(agent.execute(
#        "1.计算 12 和 24 的阶乘差多少 2. 上一步结果加上 7 等于多少 3、根据上一���结果写一首诗"
#        ).response)

    #print(llmBlock.execute("""
    #1. 获取https://huggingface.co/papers页面内容
    #2. 从页面内容中提取出所有论文的标题和链接，格式为： - 标题1: 链接1\n- 标题2: 链接2\n- 标题3: 链接3\n
    #""").response)

#    print(llmBlock.execute("""
#    1. 获取https://huggingface.co/papers/2408.14608页面内容
#    2. 输出内容中 pdf 链接的相关内容
#    """).response)

#    print(llmBlock.execute("""
#    1. #CODE 从 https://arxiv.org/pdf/2408.14608.pdf 下载文件, 并且返回下载文件地址
#    2. 读取 {resp.1} 中的内容
#    3. {resp.2}, 对上一步中的内容用中文进行总结,分为"问题概要"、"问题分析"、"问题解决"三部分阐述
#    4. #CODE 输出下面内容写到 summary.txt 文件中 -> {resp.3}
#    """).response)
#
    links = llmBlock.execute("""
        1. 获取 https://huggingface.co/papers 页面内容
        2. 页面内容如下：--{resp.1}--。从页面内容中提取出所有论文的标题和链接，格式为： -- 标题1~~链接1||-- 标题2~~链接2||-- 标题3~~链接3
    """).response

    import pdb; pdb.set_trace()

    paper_list = []
    papers = links.split("||")
    for paper in papers:
        if "~~" in paper:
            title, link = paper.split("~~")
            title = title.strip("-- ").strip()
            link = link.strip()
            paper_list.append((title, link))

    print(paper_list)