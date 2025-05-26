from milkie.agent.memory.base_memory import BaseMemory
from milkie.agent.memory.history import History
from milkie.config.config import KnowhowConfig, StorageConfig
from milkie.global_context import GlobalContext
from milkie.llm.step_llm import StepLLM
from milkie.response import Response
from milkie.utils.data_utils import extractJsonBlock

class StepLLMExtractKnowhow(StepLLM):
    def __init__(
            self, 
            globalContext: GlobalContext):
        super().__init__(
            globalContext=globalContext, 
            llm=globalContext.settings.getLLMCode())

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs) -> str:
        return f'''
            请从下面的对话记录中提取出有用的经验知识，以便后续复用。

            对话记录：
            {kwargs["history"].reprWholeDialogue()}

            每条经验知识包括两个核心字段：
            - knowhow: 经验知识
            - confidence: 置信度，取值范围为0-100

            输出格式为 jsonl，例如：
            ```
            [
                {{
                    "knowhow": "经验知识1",
                    "confidence": 95
                }},
                {{
                    "knowhow": "经验知识2",
                    "confidence": 55
                }}
            ]
            ```
            
            现在请直接输出 jsonl:
        '''

    def formatResult(self, result: Response, **kwargs):
        return extractJsonBlock(result.respStr)

class Knowhow(BaseMemory):
    def __init__(
            self, 
            storageConfig :StorageConfig,
            knowhowConfig :KnowhowConfig,
            globalContext :GlobalContext):
        super().__init__(
            type="knowhow",
            storageConfig=storageConfig,
            globalContext=globalContext)

        self.knowhowConfig = knowhowConfig
        self.stepLLMExtractKnowhow = StepLLMExtractKnowhow(globalContext)
        self.knowhows :list[dict] = self.load()

    def extract(self, history :History):
        knowhows = self.stepLLMExtractKnowhow.completionAndFormat(
            history=history)
        for knowhow in knowhows:
            self.knowhows.append(knowhow)
        self._compress()
        self.flush(self.knowhows)

    def get(self) -> list[dict]:
        return self.knowhows[:len(self.knowhows)//2] if len(self.knowhows) > 1 else self.knowhows

    def _compress(self):
        self.knowhows.sort(key=lambda x: x["confidence"], reverse=True)
        if len(self.knowhows) > self.knowhowConfig.maxNum:
            self.knowhows = self.knowhows[:self.knowhowConfig.maxNum]