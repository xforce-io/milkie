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

from abc import abstractmethod
import json
import logging
from typing import Any, Callable, List, Optional, Tuple, Dict

from milkie.config.constant import MaxLenLog
from milkie.context import VarDict
from milkie.log import INFO
from milkie.utils.data_utils import extractFromBlock, restoreVariablesInDict, unescape

from ..openai_function import OpenAIFunction
from milkie.functions.code_interpreter import CodeInterpreter

logger = logging.getLogger(__name__)

class FuncExecRecord:
    def __init__(
            self, 
            toolCall: Tuple[str, dict], 
            tool: OpenAIFunction,
            result: Any):
        self.toolCall = toolCall
        self.tool = tool
        self.result = result

    def __str__(self) -> str:
        if self.toolCall:
            return f"""
            ```
            function_name: {self.toolCall[0]}
            function_description: {self.tool.get_function_description()}
            arguments: {self.toolCall[1]}
            result: {self.result}
            ```
            """
        else:
            return f"tool: {self.tool}"

class Toolkit():
    def __init__(self, globalContext=None) -> None:
        self.records = []
        self.globalContext = globalContext
        if globalContext:
            self.codeInterpreter = CodeInterpreter(self.globalContext)

    @abstractmethod
    def getName(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def getTools(self) -> List[OpenAIFunction]:
        raise NotImplementedError("Subclasses must implement this method.")

    def getCertainTools(self, toolNames: List[str]) -> List[OpenAIFunction]:
        return [tool for tool in self.getTools() if tool.get_function_name() in toolNames]

    @staticmethod
    def getToolsWithSingleFunc(func :Callable) -> List[OpenAIFunction]:
        return [OpenAIFunction(func)]

    @staticmethod
    def getToolsDescWithSingleFunc(func :Callable) -> str:
        return Toolkit.getDescFromFunctions(Toolkit.getToolsWithSingleFunc(func))

    def getToolsSchema(self) -> list:
        return [tool.get_openai_tool_schema() for tool in self.getTools()]

    def getToolsSchemaForCertainTools(self, toolNames: List[str]) -> list:
        return [tool.get_openai_tool_schema() for tool in self.getCertainTools(toolNames)]

    def getToolsDict(self) -> dict:
        return {tool.get_function_name(): tool for tool in self.getTools()}

    def getDesc(self) -> str:
        return self.getDescFromFunctions(self.getTools())

    def getToolDescs(self) -> dict[str, str]:
        return {tool.get_function_name(): tool.get_function_description() for tool in self.getTools()}

    @staticmethod
    def getUnionToolkit(toolkits: List[Toolkit]) -> Toolkit:
        if len(toolkits) == 0:
            return None
        elif len(toolkits) == 1:
            return toolkits[0]
        
        newToolkit = toolkits[0]
        for toolkit in toolkits[1:]:
            newToolkit.getTools().extend(toolkit.getTools())
        return newToolkit

    @staticmethod
    def getDescFromFunctions(tools :List[OpenAIFunction]) -> str:
        toolDescriptions = [tool.get_function_name() + " | " + tool.get_function_description() for tool in tools]
        return "\n".join(toolDescriptions)

    def isEmpty(self) -> bool:
        return len(self.getTools()) == 0

    def extractToolFromMsg(
            self, 
            msg: str, 
            varDict :VarDict,
            needToParse :bool = False) -> List[FuncExecRecord]:
        if not msg.startswith("```json"):
            return None

        msg = unescape(extractFromBlock("json", msg))
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
            toolCalls: List[tuple], 
            allDict: dict,
            needToParse :bool = False) -> List[FuncExecRecord]:
        records = []
        for toolCall in toolCalls:
            args = json.loads(unescape(toolCall[1]))
            record = self.execFromJson(
                toolCall[0], 
                args, 
                allDict, 
                needToParse=needToParse)
            records.append(record)
        self.records.extend(records)
        return records

    def execFromJson(
            self, 
            funcName: str, 
            args: dict, 
            allDict: dict,
            needToParse :bool = False,
            **kwargs) -> FuncExecRecord:
        import pdb; pdb.set_trace()
        tool = self.getToolsDict()[funcName]
        if needToParse:
            args = restoreVariablesInDict(args, allDict)
        result = tool.func(**args, **kwargs)
        if not result or result.strip() == "":
            raise ValueError(f"funcCall func[{funcName}] args[{args}] result[{result}]")

        INFO(logger, f"funcCall func[{funcName}] args[{args}] result[{result[:MaxLenLog]}]")
        return FuncExecRecord((funcName, args), tool, result)

    def genCodeAndRun(self, instruction: str, varDict: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        r"""根据指令生成代码，并且用代码解释器执行代码。

        Args:
            instruction (str): 要执行的指令。

        Returns: 执行结果
        """
        return self.codeInterpreter.execute(instruction, varDict=varDict, **kwargs)

    def runCode(self, code: str, varDict: Optional[Dict[str, Any]] = None) -> Any:
        r"""直接执行代码解释器

        Args:
            code (str): 要执行的代码。

        Returns: 执行结果
        """
        return self.codeInterpreter.executeCode(code, varDict)

class EmptyToolkit(Toolkit):

    def __init__(self, globalContext=None) -> None:
        super().__init__(globalContext)

    def getTools(self) -> List[OpenAIFunction]:
        return []
