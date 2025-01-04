import re, logging
from enum import Enum

from llama_index.core import Response

from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.llm.inference import chat
from milkie.tools.tool import LLM, Coder, Tool, ToolSet

logger = logging.getLogger(__name__)

class RoleAgent(BaseBlock):
    
    def __init__(
            self, 
            role: str,
            goal: str,
            backstory: str,
            toolSet :ToolSet,
            planDesc :str,
            context: Context = None, 
            config: str = None,
            ) -> None:
        super().__init__(context, config)

        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.toolSet = toolSet
        self.planDesc = planDesc
        self.systemPrompt = f"""
        角色: {self.role}
        目标: {self.goal}
        背景: {self.backstory}
        """

        context.reqTrace.set("role", self.role)
        context.reqTrace.set("goal", self.goal)
        context.reqTrace.set("tools", self.toolSet.describe())
    
    def getSettings(self):
        return self.context.settings
    
    def execute(self, query: str, args :dict, **kwargs) -> Response:
        self.taskEngine = TaskEngine(self, query=query)
        resp = self.taskEngine.execute()
        logger.info(f"RoleAgent[{self.role}] exelog[{self.context.reqTrace.dump()}]")
        return resp

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
        TOOL = 1
        DECOMPOSE = 2
        CANNOT = 3
        ERROR = 4

    def __init__(
            self,
            result: Result,
            tool: Tool=None,
            instructions: list=None):
        self.result = result
        self.tool = tool
        self.instructions = instructions

class PromptMaker:
    
    def __init__(self) -> None:
        self.task :str = None
        self.curTask :str = None
        self.details :str = None
        self.thinking :str = None
        self.toolsDesc :str = None
        self.optionDecompose = False
        self.prev = None

    def addTask(self, task :str):
        self.task = task

    def addCurTask(self, curTask :str):
        self.curTask = curTask

    def addDetails(self, details :str):
        self.details = details

    def addPrev(self, prev):
        self.prev = prev

    def addThinking(self, thinking :str):
        self.thinking = thinking

    def addToolsDesc(self, toolsDesc :str):
        self.toolsDesc = toolsDesc

    def addOptionDecompose(self):
        self.optionDecompose = True

    def makeAnswerPrompt(self):
        resultPrompt = self._promptTask()
        
        prevResp = self._getPrevObservation()
        if prevResp:
            resultPrompt += f"""
        上个任务执行结果：
        --------------------------------
        {prevResp}
        --------------------------------
            """

        resultPrompt += f"""
        请尝试根据上面的任务执行结果等信息，对任务给出答案:

        - 如果任务已经完成，请直接给出答案，返回格式如下
        --------------------------------
        任务回答：XXX
        --------------------------------

        - 如果任务尚未完成，或者根据目前的执行结果无法判断，请返回如下格式
        --------------------------------
        无法直接回答
        --------------------------------

        现在请给出结果：
        """
        return resultPrompt

    def makeThinkingPrompt(self):
        resultPrompt = self._promptTask()

        prevResp = self._getPrevObservation()
        if prevResp:
            resultPrompt += f"""
        上个任务执行结果：
        --------------------------------
        {prevResp}
        --------------------------------
            """

        resultPrompt += f"""
        请对上面的请求给出自己解决问题的思路，请用一整段文字进行陈述，千万不需要分解为具体的步骤，格式如下：
        --------------------------------
        解决思路：XXX
        --------------------------------

        现在请给出结果：
        """
        return resultPrompt

    def makeDecisionPrompt(self):
        resultPrompt = self._promptTask()
        
        if self.thinking:
            resultPrompt += f"""
        目前的思考结果：
        --------------------------------
        {self.thinking}
        --------------------------------
            """

        prevResp = self._getPrevObservation()
        if prevResp:
            resultPrompt += f"""
        上个任务执行结果：
        --------------------------------
        {prevResp}
        --------------------------------
            """

        if self.toolsDesc:
            resultPrompt += f"""
        我们拥有的工具包括: 
        --------------------------------
        {self.toolsDesc}
        --------------------------------
            """ 

        resultPrompt += f"""
        基于上面的思考结果，现在我们需要做出分析，并且做出第一步判断:
        """

        if self.optionDecompose:
            resultPrompt += f"""
        如果不能直接回答或使用工具解决，判断该任务是否需要进一步拆解来进行解决。
            """

        resultPrompt += f"""
        如果不能，判断该任务无法解决。
        """

        resultPrompt += f"""
        - 如果认为该任务可以使用工具解决，请选择工具，返回格式如下
        --------------------------------
        分析结果：使用工具
        工具名称：XXX
        --------------------------------
        """

        if self.optionDecompose:
            resultPrompt += f"""
        - 如果认为该任务需要进一步拆解，请结合任务描述和思考结果对任务进行进一步拆解，返回格式如下
        --------------------------------
        分析结果：需进一步拆解
        1. **步骤 1**
        - XXX
        - XXX
        
        2. **步骤 2**
        - XXX
        - XXX
        ...
        --------------------------------
            """
            
        resultPrompt += f"""
        - 如果认为该任务无法解决，返回格式如下
        --------------------------------
        分析结果：无法解决
        原因：XXX
        --------------------------------
        
        现在请给出分析结果:
        """

        return resultPrompt

    def makeToolusePrompt(self):
        resultPrompt = self._promptTask()

        if self.thinking:
            resultPrompt += f"""
        目前的思考结果：
        --------------------------------
        {self.thinking}
        --------------------------------
            """
        
        prevResp = self._getPrevObservation()
        if prevResp:
            resultPrompt += f"""
        上个任务执行结果：
        --------------------------------
        {prevResp}
        --------------------------------
            """
        return resultPrompt

    def _getPrevObservation(self):
        return self.prev.observation if self.prev else None

    def _promptTask(self, withGeneralTask=True):
        resultPrompt = f"""
        任务：
        --------------------------------
        {self.curTask}
        --------------------------------
        """

        if self.details:
            resultPrompt += f"""
        任务详情：
        --------------------------------
        {self.details}
        --------------------------------
            """

        if withGeneralTask and self.task != self.curTask:
            resultPrompt += f"""
            总体目标：
            --------------------------------
            {self.task}
            --------------------------------
            """
        return resultPrompt

class InstructResult:
    def __init__(
            self, 
            response :Response,
            isEnd :bool=False) -> None:
        self.response = response
        self.isEnd = isEnd

class Instruction:
    def __init__(
            self, 
            roleAgent: RoleAgent, 
            task :str,
            curTask :str,
            details :list=None,
            observation :str=None,
            prev=None,
            level :int = 1) -> None:
        self.roleAgent = roleAgent
        self.task = task
        self.curTask = curTask
        self.details = details
        self.prev :Instruction = prev

        self.thinkingResult :Response = None
        self.analysisResult :Response = None

        self.result :Response = None

        self.observation :str = observation

        self.level = level

        self.promptMaker = PromptMaker()
        self.promptMaker.addTask(self.task)
        self.promptMaker.addCurTask(self.curTask)
        self.promptMaker.addDetails(self.details)
        self.promptMaker.addPrev(self.prev)
        self.promptMaker.addToolsDesc(
            self.roleAgent.toolSet.describe())
        if self.level == 1:
            self.promptMaker.addOptionDecompose()

    def execute(self, instructions :list) -> InstructResult:
        if self._getPrevObservation():
            self.answerResult = self.answer()
            if self.answerResult is None:
                return None

            answerResult = self._formatAnswerResult(
                self.answerResult.response)
            if answerResult.result == AnswerResult.Result.ANSWER:
                return InstructResult(
                    Response(response=answerResult.response),
                    True)
        
        self.thinkingResult = self._think()
        if self.thinkingResult is None:
            logger.error("no thinking result")
            return None
        
        thinkingResult = self._formatThinkingResult(
            self.thinkingResult.response)
        if thinkingResult.result == ThinkingResult.Result.THINK:
            self.promptMaker.addThinking(thinkingResult.response)
        else:
            logger.error("fail parse thinking result")
            return None

        self.analysisResult = self._analysis()
        if self.analysisResult is None:
            logger.error("no analysis result")
            return None
        
        log = {
            "thinking" : self.thinkingResult.response,
            "analysis" : self.analysisResult.response,
        }
        self.roleAgent.context.reqTrace.add("instruction", log)

        analysisResult = self._formatAnalysisResult(
            self.analysisResult.response)
        if analysisResult.result == AnalysisResult.Result.TOOL:
            self.result = self._useTool(
                analysisResult.tool,
                thinkingResult.response)
            if self.result and self.result.response:
                self.observation = self.result.response
        elif analysisResult.result == AnalysisResult.Result.DECOMPOSE:
            analysisResult.instructions.reverse()
            for inst in analysisResult.instructions:
                instructions.insert(0, inst)
            self.result = Response(response="decompose")
        return InstructResult(self.result)

    def answer(self) -> Response:
        prompt = self.promptMaker.makeAnswerPrompt()
        response = chat(
            llm=self.roleAgent.context.globalContext.settings.llm, 
            systemPrompt=self.roleAgent.systemPrompt,
            prompt=prompt, 
            promptArgs={})
        return response

    def _getPrevObservation(self):
        return self.prev.observation if self.prev else None

    def _think(self) -> Response:
        prompt = self.promptMaker.makeThinkingPrompt()
        response = chat(
            llm=self.roleAgent.context.globalContext.settings.llm, 
            systemPrompt=self.roleAgent.systemPrompt,
            prompt=prompt, 
            promptArgs={})
        return response

    def _analysis(self) -> Response:
        prompt = self.promptMaker.makeDecisionPrompt()
        response = chat(
            llm=self.roleAgent.context.globalContext.settings.llm, 
            systemPrompt=self.roleAgent.systemPrompt,
            prompt=prompt, 
            promptArgs={})
        return response

    def _formatAnswerResult(
            self,
            answerResult: str) -> AnswerResult:
        if "任务回答：" in answerResult:
            pattern = re.compile(r'任务回答：(.+)\n?')
            match = pattern.findall(answerResult)
            if match:
                return AnswerResult(
                    AnswerResult.Result.ANSWER,
                    response=match[0]
                )            
        return AnswerResult(
            AnswerResult.Result.NOANS,
            response=None)

    def _formatThinkingResult(
            self,
            thinkingResult: str) -> ThinkingResult:
        if "解决思路：" in thinkingResult:
            pattern = re.compile(r'解决思路：(.+)\n?')
            match = pattern.findall(thinkingResult)
            if match:
                return ThinkingResult(
                    ThinkingResult.Result.THINK,
                    response=match[0]
                )
        return ThinkingResult(
            ThinkingResult.Result.ERROR,
            response=None)

    def _formatAnalysisResult(
            self, 
            analysisResult: str) -> AnalysisResult:
        if "分析结果：使用工具" in  analysisResult:
            pattern = re.compile(r'工具名称：(.+)\n?')
            match = pattern.findall(analysisResult)
            if match:
                if match[0].lower() == LLM().name():
                    return AnalysisResult(
                        AnalysisResult.Result.TOOL,
                        tool=LLM()
                    )
                elif match[0].lower() == Coder().name():
                    return AnalysisResult(
                        AnalysisResult.Result.TOOL,
                        tool=Coder()
                    )
            return AnalysisResult(AnalysisResult.Result.ERROR)
        elif "分析结果：需进一步拆解" in analysisResult:
            instructions = self._decomposePlan(analysisResult)
            return AnalysisResult(
                AnalysisResult.Result.DECOMPOSE,
                instructions=instructions
            )
        elif "分析结果：无法解决" in analysisResult:
            return AnalysisResult(AnalysisResult.Result.CANNOT)
        return AnalysisResult(AnalysisResult.Result.ERROR)

    def _decomposePlan(self, planDesc :str):
        pattern = re.compile(r'\d+\.\s\*\*(.*?)\*\*\n((\s*-\s+.*\n)*)')
        matches = pattern.findall(planDesc)
        instructions = []
        lastInstruction :Instruction = None
        for match in matches:
            stepTitle = match[0].strip()
            stepDetails = [detail.strip() for detail in match[1].strip().split('\n') if detail.strip()]
            curInstruction = Instruction(
                roleAgent=self.roleAgent, 
                task=self.task,
                curTask=stepTitle, 
                details=stepDetails,
                prev=lastInstruction,
                level=self.level + 1)
            instructions.append(curInstruction)
            lastInstruction = curInstruction
        return instructions

    def _useTool(self, tool :Tool, thinking :str):
        prompt = self.promptMaker.makeToolusePrompt()
        return tool.execute(self.roleAgent.context, prompt)

class TaskEngine:
    def __init__(
            self, 
            roleAgent :RoleAgent,
            query :str) -> None:
        self.roleAgent = roleAgent
        self.query = query
        self.lastInstruction :Instruction = None

    def execute(self) -> Response:
        result = self._executeOnce()
        if result[0] or not self.lastInstruction:
            return result[1]

        return self._executeOnce()[1]

    def _executeOnce(self) -> tuple:
        mainInstruction = Instruction(
            roleAgent=roleAgent, 
            task=self.query, 
            curTask=self.query,
            prev=self.lastInstruction)

        self.instructions = [mainInstruction]
        while len(self.instructions) != 0:
            instructResult = self._step()
            if not instructResult:
                raise ValueError(f"TaskEngine failed to execute")
            
            if instructResult.isEnd:
                return (True, instructResult.response)
        return (False, instructResult.response)

    def _step(self) -> InstructResult:
        if len(self.instructions) == 0:
            return

        instruction = self.instructions.pop(0)
        self.lastInstruction = instruction
        return instruction.execute(self.instructions)

if __name__ == "__main__":
    roleAgent = RoleAgent(
        role="你是一个仔细认真的数据分析师，需要使用各种提供的工具进行数据分析，来解决问题",
        goal="快速分析和洞察数据，提供有用的见解",
        backstory=None,
        toolSet=ToolSet([Coder(), LLM()]),
        planDesc=None)

    response = roleAgent.execute("data路径下有个文件名包含‘南区技术’的excel文件，帮我找出'最终'列得分最高的前三人是谁，他们分别是什么时候入职的以及 base 在哪里。如果你不了解需要利用哪些列和字段进行计算，可以先阅读下文件的头几行了解下数据格式", args={})
    #response = roleAgent.execute("data路径下有个文件名包含‘南区技术’的excel文件，帮我总结下这个表格的内容，得出一些分析结果", args={})
    #response = roleAgent.execute("我该怎么做西红柿炒蛋", args={})
    print(response.response)