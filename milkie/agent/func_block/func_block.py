from __future__ import annotations
import re
from typing import List
from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.config.constant import InstFlagFunc, KeywordForStart
import logging

from milkie.trace import stdout
from milkie.utils.data_utils import restoreVariablesInDict, restoreVariablesInStr

logger = logging.getLogger(__name__)

class FuncBlock(BaseBlock):
    def __init__(
            self,
            agentName: str,
            funcDefinition: str=None,
            funcName: str = None,
            params: List[str] = [],
            globalContext: GlobalContext = None,
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None  # 添加 repoFuncs 参数
    ):
        super().__init__(
            agentName=agentName, 
            globalContext=globalContext, 
            config=config, 
            toolkit=toolkit, 
            repoFuncs=repoFuncs
        )

        if funcDefinition:
            self.funcDefinition = funcDefinition.strip()
            self.funcName = None
            self.params = []
        else:
            self.funcDefinition = None
            self.funcName = funcName
            self.params = params

        self.flowBlock = None
        self.funcCallPattern :str = None

    def getFuncNamePattern(self):
        return f"{InstFlagFunc}{self.funcName}"

    def setFuncCallPattern(self, funcCallPattern: str):
        self.funcCallPattern = funcCallPattern

    def getFuncCallPattern(self):
        return self.funcCallPattern

    def compile(self):
        if not self.funcDefinition:
            return

        lines = self.funcDefinition.strip().split('\n')
        self.parseFunctionDefinition(lines)

        # Create FlowBlock
        from milkie.agent.flow_block import FlowBlock
        self.flowBlock = FlowBlock.create(
            agentName=self.agentName,
            globalContext=self.globalContext,
            config=self.config,
            toolkit=self.toolkit,
            flowCode=self.flowBlockCode,
            repoFuncs=self.repoFuncs  # 传递 repoFuncs 给 FlowBlock
        )
        self.flowBlock.compile()

    def parseFunctionDefinition(self, lines):
        state = 'start'
        nestingLevel = 0
        flowLines = []

        for line in lines:
            stripped = line.strip()
            
            if state == 'start' and stripped.startswith('DEF'):
                # Parse function name and parameters
                match = re.match(r'DEF\s+(\w+)\((.*?)\)', stripped)
                if not match:
                    raise SyntaxError("Invalid function definition syntax")
                self.funcName = match.group(1)
                self.params = [param.strip() for param in match.group(2).split(',')]
                logger.debug(f"Parsed function: {self.funcName}, params: {self.params}")
                state = 'body'
                nestingLevel = 1
            elif state == 'body':
                if stripped.startswith('END'):
                    nestingLevel -= 1
                    if nestingLevel == 0:
                        break
                elif any(stripped.startswith(keyword) for keyword in ['FOR']):
                    nestingLevel += 1
                flowLines.append(line)

        if nestingLevel != 0:
            raise SyntaxError("Incomplete function definition")

        if len(flowLines) == 0:
            raise SyntaxError("No function body found")

        self.flowBlockCode = '\n'.join(flowLines)
        logger.debug(f"Function body parsed: {self.flowBlockCode[:50]}...")

    def processNestedFor(self, code):
        lines = code.split('\n')
        processedLines = []
        forStack = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(KeywordForStart):
                forStack.append(line)
            elif stripped == 'END' and forStack:
                forStack.append(line)
                if len(forStack) == 1:
                    processedLines.append('\n'.join(forStack))
                    forStack.pop()
            elif forStack:
                forStack.append(line)
            else:
                processedLines.append(line)
        
        return '\n'.join(processedLines)

    def setParams(self, args :List[str]):
        if len(args) > len(self.params):
            raise ValueError(f"Expected arguments[{self.params}], but got[{args}]")

        if len(args) < len(self.params):
            args = args + [None] * (len(self.params) - len(args))

        for param, value in zip(self.params, args):
            self.setVarDictDataSegment(param, value)

    def execute(
            self, 
            context: Context,
            args: dict = {}, 
            prevBlock: BaseBlock = None, 
            **kwargs) -> Response:
        super().execute(
            context=context, 
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)
        
        params = self._restoreParams(args, self.params)

        stdout(f"called func start: {self.funcName}, params: {params}", **kwargs)
        response = self.flowBlock.execute(
            context=context,
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)
        stdout(f"called func end: {self.funcName}", **kwargs)

        self.clearVarDictLocal(self.params)
        return response

    def _restoreParams(self, args :dict, params :List[str]) -> dict:
        replaced = {}
        for param, value in self.getVarDict().getLocalDict().items():
            if param not in params:
                continue

            if type(value) == str:
                replaced[param] = restoreVariablesInStr(value, self.getVarDict().getGlobalDict())
            elif type(value) == dict:
                replaced[param] = restoreVariablesInDict(value, self.getVarDict().getGlobalDict())
            elif type(value) == list:
                replaced[param] = value
            else:
                raise ValueError(f"Unsupported type: {type(value)}")

            args[param] = replaced[param]
        
        for param, value in replaced.items():
            self.setVarDictLocal(param, value)
        return replaced

    def __str__(self):
        return f"FuncBlock(funcName={self.funcName}, params={self.params})"

    @staticmethod
    def create(
            agentName: str,
            funcDefinition: str,
            context: Context = None,
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None) -> 'FuncBlock':
        return FuncBlock(
            agentName=agentName,
            funcDefinition=funcDefinition,
            context=context,
            config=config,
            toolkit=toolkit,
            repoFuncs=repoFuncs
        )

class RepoFuncs:
    def __init__(self):
        self.funcs = []

    def add(self, name: str, funcBlock: FuncBlock):
        self.funcs.append((name, funcBlock))

    def get(self, name: str) -> FuncBlock:
        for funcName, funcBlock in self.funcs:
            if funcName == name:
                return funcBlock
        return None

    def getAll(self) -> dict:
        return {name: funcBlock for name, funcBlock in self.funcs}

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        return iter(self.funcs)