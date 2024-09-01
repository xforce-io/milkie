from __future__ import annotations

# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import json
import logging
from typing import Any, List

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from milkie.config.constant import MaxLenLogField

from .openai_function import OpenAIFunction

logger = logging.getLogger(__name__)

class FuncExecRecord:
    def __init__(
            self, 
            toolCall: ChatCompletionMessageToolCall, 
            tool: OpenAIFunction,
            result: Any):
        self.toolCall = toolCall
        self.tool = tool
        self.result = result

    def __str__(self) -> str:
        return f"""
        ```
        function_name: {self.toolCall.function.name}
        function_description: {self.tool.get_function_description()}
        arguments: {self.toolCall.function.arguments}
        result: {self.result}
        ```
        """

class BaseToolkit():

    def __init__(self) -> None:
        self.records = []

    def getTools(self) -> List[OpenAIFunction]:
        raise NotImplementedError("Subclasses must implement this method.")

    def getToolsSchema(self) -> list:
        return [tool.get_openai_tool_schema() for tool in self.getTools()]

    def getToolsDict(self) -> dict:
        return {tool.get_function_name(): tool for tool in self.getTools()}

    def getToolsDesc(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def exec(self, toolCalls: List[ChatCompletionMessageToolCall]) -> List[FuncExecRecord]:
        records = []
        for toolCall in toolCalls:
            tool = self.getToolsDict()[toolCall.function.name]
            args = json.loads(toolCall.function.arguments)
            result = tool.func(**args)
            logger.info(f"funcCall func[{toolCall.function.name}] args[{toolCall.function.arguments}] result[{result[:MaxLenLogField]}]")
            
            record = FuncExecRecord(toolCall, tool, result)
            records.append(record)
        self.records.extend(records)
        return records