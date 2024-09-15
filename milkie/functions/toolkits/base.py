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
import re
from typing import Any, Callable, List, Tuple, Dict

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from milkie.config.constant import MaxLenLog
from milkie.utils.data_utils import restoreVariablesInDict

from ..openai_function import OpenAIFunction

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

    @staticmethod
    def getToolsWithSingleFunc(func :Callable) -> List[OpenAIFunction]:
        return [OpenAIFunction(func)]

    @staticmethod
    def getToolsDescWithSingleFunc(func :Callable) -> str:
        return BaseToolkit.getDesc(BaseToolkit.getToolsWithSingleFunc(func))

    def getToolsSchema(self) -> list:
        return [tool.get_openai_tool_schema() for tool in self.getTools()]

    def getToolsDict(self) -> dict:
        return {tool.get_function_name(): tool for tool in self.getTools()}

    def getToolsDesc(self) -> str:
        return self.getDesc(self.getTools())

    @staticmethod
    def getDesc(tools :List[OpenAIFunction]) -> str:
        toolDescriptions = [tool.get_function_name() + " | " + tool.get_function_description() for tool in tools]
        return "\n".join(toolDescriptions)

    def extractToolFromMsg(
            self, 
            msg: str, 
            varDict :dict,
            needToParse :bool = False) -> List[FuncExecRecord]:
        if not msg.startswith("```json"):
            return None

        msg = re.sub(r'```json\s*|\s*```', '', msg)
        msg = msg.replace("{{", "{").replace("}}", "}")
        try:
            data = json.loads(msg)
            funcName = data.get('name')
            if not funcName in self.getToolsDict():
                return None
            
            parameters = data.get('parameters', {})
            tool = self.getToolsDict()[funcName]
            if needToParse:
                parameters = restoreVariablesInDict(parameters, varDict)
            result = tool.func(**parameters)
            return [FuncExecRecord(None, tool, result)]
        except json.JSONDecodeError:
            return None
        except Exception as e:
           raise ValueError(f"Error extracting tool from message: {str(e)}")

    def exec(
            self, 
            toolCalls: List[ChatCompletionMessageToolCall], 
            varDict: dict,
            needToParse :bool = False) -> List[FuncExecRecord]:
        records = []
        for toolCall in toolCalls:
            arguments = toolCall.function.arguments
            arguments = arguments.replace("{{", "{").replace("}}", "}")
            args = json.loads(arguments)
            record = self.execFromJson(
                toolCall.function.name, 
                args, 
                varDict, 
                toolCall,
                needToParse=needToParse)
            records.append(record)
        self.records.extend(records)
        return records

    def execFromJson(
            self, 
            funcName: str, 
            args: dict, 
            varDict: dict,
            toolCall: Any,
            needToParse :bool = False) -> FuncExecRecord:
        tool = self.getToolsDict()[funcName]
        if needToParse:
            args = restoreVariablesInDict(args, varDict)
        result = tool.func(**args)
        logger.info(f"funcCall func[{funcName}] args[{args}] result[{result[:MaxLenLog]}]")
        return FuncExecRecord(toolCall, tool, result)