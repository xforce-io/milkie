from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.step_llm import StepLLM
from milkie.global_context import GlobalContext
from milkie.response import Response

class StepLLMExtractor(StepLLM):
    def __init__(
            self, 
            globalContext: GlobalContext):
        super().__init__(
            globalContext=globalContext, 
            llm=globalContext.settings.getLLMCode())

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs) -> str:
        return f'''
            任务：从给定文本中提取[{args["toExtract"]}]。

            输入文本：
            """
            {args["text"]}
            """

            指令：
            1. 仔细分析输入文本，提取与指定类型相关的信息。
            2. 如果成功提取到信息，直接返回提取的内容，不要添加任何额外的文字。
            3. 如果无法提取到相关信息，请直接返回"无相关信息"。

            输出要求：
            - 只返回提取的信息或"无相关信息"
            - 不要包含任何解释、前缀、后缀或额外的标点符号

            请按照以上指令和要求处理给定的输入文本，结果是：
        '''

    def formatResult(self, result: Response, **kwargs):
        return result.respStr