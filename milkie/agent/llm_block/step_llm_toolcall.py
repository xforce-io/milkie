import json
from milkie.llm.step_llm import StepLLM
from milkie.global_context import GlobalContext
from milkie.response import Response

class StepLLMToolCall(StepLLM):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(
            globalContext=globalContext, 
            promptMaker=None,
            llm=globalContext.settings.getLLMCode())

    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        return "你是一个助手,需要判断是否需要使用工具来处理用户的输入"

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs) -> str:
        return kwargs["query"]

    def formatResult(self, result: Response, **kwargs) -> list:
        messages = result.getChoicesMessages()
        tool_calls = []
        for message in messages:
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_calls.append({
                    "need_tool": True,
                    "tool_name": tool_call.function.name,
                    "tool_args": json.loads(tool_call.function.arguments)
                })
        return tool_calls