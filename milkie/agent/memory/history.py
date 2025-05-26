from typing import List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import copy

class History:
    """对话历史管理类"""
    def __init__(self):
        self.systemPrompt: Optional[str] = None
        self.history: List[ChatMessage] = []
        self.resetUse()

    def resetUse(self) -> None:
        self.activate = True

    def use(self) -> bool:
        """历史记录在单个agent运行中只能使用一次"""
        value = self.activate
        self.activate = False
        return value

    def setSystemPrompt(self, systemPrompt: str) -> None:
        self.systemPrompt = systemPrompt

    def addUserPrompt(self, userPrompt: str) -> None:
        self.history.append(ChatMessage(content=userPrompt, role=MessageRole.USER))

    def addAssistantPrompt(self, assistantPrompt: str) -> None:
        self.history.append(ChatMessage(content=assistantPrompt, role=MessageRole.ASSISTANT))

    def getDialogue(self) -> List[ChatMessage]:
        """获取完整对话历史，包括系统提示"""
        if self.systemPrompt:
            return [ChatMessage(content=self.systemPrompt, role=MessageRole.SYSTEM)] + self.history
        return self.history

    def getRecentDialogue(self) -> List[ChatMessage]:
        """获取最近一次对话"""
        if self.systemPrompt:
            return [ChatMessage(content=self.systemPrompt, role=MessageRole.SYSTEM), self.history[-1]]
        return [self.history[-1]]

    def getRecentUserPrompt(self) -> Optional[str]:
        """获取最近一次用户对话"""
        return self.history[-1].content if len(self.history) > 0 else None

    def copy(self) -> 'History':
        return copy.deepcopy(self)

    def reprWholeDialogue(self) -> str:
        reprDialogue = f"systemPrompt: {self.systemPrompt}\n"
        for message in self.history:
            reprDialogue += f"{message.role}: {message.content}\n"
        return reprDialogue

from llama_index.core import ChatPromptTemplate

def makeMessageTemplates(
        systemPrompt :str, 
        history :Optional[History], 
        prompt :str,
        reasoningModel :bool) -> ChatPromptTemplate:
    messageTemplates = []
    if history and history.use():
        history.setSystemPrompt(systemPrompt)
        history.addUserPrompt(prompt)
        messageTemplates = history.getRecentDialogue()
    else:
        if systemPrompt and not reasoningModel:
            messageTemplates += [ChatMessage(content=systemPrompt, role=MessageRole.SYSTEM)]    
        messageTemplates += [ChatMessage(content=prompt, role=MessageRole.USER)]
    return ChatPromptTemplate(message_templates=messageTemplates)
