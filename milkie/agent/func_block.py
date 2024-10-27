from __future__ import annotations
import re
from typing import List
from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.response import Response
from milkie.config.constant import InstFlagFunc, KeywordForStart
import logging

from milkie.trace import stdout
from milkie.utils.data_utils import restoreVariablesInStr

logger = logging.getLogger(__name__)

class FuncBlock(BaseBlock):
    def __init__(
            self,
            funcDefinition: str,
            context: Context = None,
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None  # 添加 repoFuncs 参数
    ):
        super().__init__(context, config, toolkit, repoFuncs=repoFuncs)  # 传递 repoFuncs 给父类
        self.funcDefinition = funcDefinition.strip()
        self.funcName = None
        self.params = []
        self.flowBlock = None

    def getFuncPattern(self):
        return f"{InstFlagFunc}{self.funcName}"

    def compile(self):
        lines = self.funcDefinition.strip().split('\n')
        self.parseFunctionDefinition(lines)

        # Create FlowBlock
        from milkie.agent.flow_block import FlowBlock
        self.flowBlock = FlowBlock.create(
            context=self.context,
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
        if len(args) != len(self.params):
            raise ValueError(f"Expected arguments[{self.params}], but got[{args}]")

        for param, value in zip(self.params, args):
            self.setVarDictLocal(param, value)

    def execute(self, query: str = None, args: dict = {}, prevBlock: BaseBlock = None, **kwargs) -> Response:
        params = self._restoreParams(args)

        stdout(f"called func start: {self.funcName}, params: {params}", **kwargs)
        response = self.flowBlock.execute(query, args, prevBlock)
        stdout(f"called func end: {self.funcName}", **kwargs)

        self.clearVarDictLocal()
        return response

    def _restoreParams(self, args :dict) -> dict:
        replaced = {}
        for param, value in self.getVarDict().getLocalDict().items():
            replaced[param] = restoreVariablesInStr(value, self.getVarDict().getGlobalDict())
            if param in args: args[param] = replaced[param]
        
        for param, value in replaced.items():
            self.setVarDictLocal(param, value)
        return replaced

    def __str__(self):
        return f"FuncBlock(funcName={self.funcName}, params={self.params})"

    @staticmethod
    def create(
            funcDefinition: str,
            context: Context = None,
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None) -> 'FuncBlock':
        return FuncBlock(funcDefinition, context, config, toolkit, repoFuncs)

class RepoFuncs:
    def __init__(self):
        self.funcs = {}

    def add(self, name: str, func_block: FuncBlock):
        self.funcs[name] = func_block

    def get(self, name: str) -> FuncBlock:
        return self.funcs.get(name)

    def getAll(self) -> dict:
        return self.funcs

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        return iter(self.funcs.values())