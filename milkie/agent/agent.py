from typing import List, Union
from milkie.agent.base_block import BaseBlock
from milkie.agent.flow_block import FlowBlock
from milkie.agent.func_block import FuncBlock, RepoFuncs
from milkie.config.constant import KeywordFuncStart, KeywordFuncEnd
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.base_toolkits import BaseToolkit
from milkie.response import Response
import logging

logger = logging.getLogger(__name__)

class Agent(BaseBlock):
    def __init__(
            self, 
            code: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: BaseToolkit = None,
            usePrevResult=False):
        self.repoFuncs = RepoFuncs()
        super().__init__(context, config, toolkit, usePrevResult, self.repoFuncs)
        self.code = code
        self.funcBlocks: List[FuncBlock] = []
        self.flowBlocks: List[FlowBlock] = []

    def compile(self):
        lines = self.code.split('\n')
        currentBlock = []
        inFuncBlock = False
        funcBlockCount = 0
        for line in lines:
            strippedLine = line.strip()
            if len(strippedLine) == 0:
                continue

            if strippedLine.startswith(KeywordFuncStart):
                if currentBlock:
                    self.addFlowBlock(currentBlock)
                    currentBlock = []
                inFuncBlock = True
                currentBlock = [line]
                funcBlockCount += 1
                logger.debug(f"Found function definition: {strippedLine}")
            elif strippedLine == KeywordFuncEnd and inFuncBlock:
                currentBlock.append(line)
                self.addFuncBlock(currentBlock)
                currentBlock = []
                inFuncBlock = False
            else:
                currentBlock.append(line)

        if currentBlock:
            self.addFlowBlock(currentBlock)

        logger.debug(f"Total function blocks found: {funcBlockCount}")
        logger.debug(f"Total function blocks added: {len(self.repoFuncs.funcs)}")

        for block in self.funcBlocks:
            block.compile()

        for funcBlock in self.funcBlocks:
            funcName = funcBlock.funcName
            self.repoFuncs.add(funcName, funcBlock)
            logger.debug(f"Added function block: {funcName}")

        for block in self.flowBlocks:
            block.compile()

    def addFuncBlock(self, lines):
        funcBlock = FuncBlock.create(
            '\n'.join(lines),
            context=self.context,
            config=self.config,
            toolkit=self.toolkit,
            repoFuncs=self.repoFuncs
        )
        self.funcBlocks.append(funcBlock)

    def addFlowBlock(self, lines):
        self.flowBlocks.append(FlowBlock.create(
            '\n'.join(lines),
            context=self.context,
            config=self.config,
            toolkit=self.toolkit,
            usePrevResult=self.usePrevResult,
            repoFuncs=self.repoFuncs
        ))

    def execute(
            self, 
            query: str = None, 
            args: dict = {}, 
            prevBlock: BaseBlock = None) -> Response:
        result = Response()
        lastBlock = prevBlock

        for block in self.flowBlocks:
            result = block.execute(
                query=query,
                args=args,
                prevBlock=lastBlock
            )
            lastBlock = block
        return result

if __name__ == "__main__":
    code = """
    DEF down(ceiling)
        返回比{ceiling}小的最大的奇数，直接返回结果
    END

    1. 10 以内最大的质数 -> num
    2. 以@down({num})为主题写一首诗
    """
    agent = Agent(code)
    agent.compile()
    print(agent.execute().resp)
