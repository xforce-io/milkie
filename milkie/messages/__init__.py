from milkie.types import (
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
)

OpenAISystemMessage = ChatCompletionSystemMessageParam
OpenAIAssistantMessage = ChatCompletionAssistantMessageParam
OpenAIUserMessage = ChatCompletionUserMessageParam
OpenAIFunctionMessage = ChatCompletionFunctionMessageParam
OpenAIMessage = ChatCompletionMessageParam

from .base import BaseMessage  # noqa: E402
from .func_message import FunctionCallingMessage  # noqa: E402

__all__ = [
    'OpenAISystemMessage',
    'OpenAIAssistantMessage',
    'OpenAIUserMessage',
    'OpenAIMessage',
    'BaseMessage',
    'FunctionCallingMessage',
]
