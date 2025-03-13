import logging
from typing import List

from milkie.functions.openai_function import OpenAIFunction
from milkie.global_context import GlobalContext
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.tools.basic_tools import ToolSendEmail

logger = logging.getLogger(__name__)

class BasicToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

        self.toolSendEmail = ToolSendEmail(globalContext)

    def getName(self) -> str:
        return "BasicToolkit"

    def sendEmail(self, to_email: str, subject: str, body: str) -> str:
        """
        发送电子邮件给指定邮箱

        Args:
            to_email (str): 收件人邮箱地址
            subject (str): 邮件主题
            body (str): 邮件正文

        Returns:
            str: 发送结果描述

        Raises:
            Exception: 如果发送邮件过程中出现错误
        """
        return self.toolSendEmail.execute(to_email, subject, body)

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.sendEmail),
        ]
