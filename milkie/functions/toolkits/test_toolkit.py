import logging
from typing import List

from milkie.functions.openai_function import OpenAIFunction
from milkie.global_context import GlobalContext
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.tools.basic_tools import ToolSendEmail

logger = logging.getLogger(__name__)

class TestToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

    def getName(self) -> str:
        return "TestToolkit"

    def genAhaStr(self) -> str:
        """
        生成一个啊哈字符串。

        Returns:
            str: 生成的啊哈字符串

        Example:
            {
                "name": "genAhaStr",
                "description": "生成一个啊哈字符串",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        """
        import uuid
        return str(uuid.uuid4())

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.genAhaStr),
        ]

