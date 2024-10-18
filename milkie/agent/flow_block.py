from typing import List, Union
from milkie.agent.base_block import BaseBlock
from milkie.agent.for_block import ForBlock
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.config.constant import DefaultUsePrevResult, KeywordForEnd, KeywordForStart
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.response import Response

class FlowBlock(BaseBlock):
    def __init__(
            self, 
            flowCode: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            repoFuncs=None):
        super().__init__(context, config, toolkit, usePrevResult, repoFuncs)
        self.flowCode = flowCode
        self.blocks: List[Union[LLMBlock, ForBlock]] = []

    def compile(self):
        lines = self.flowCode.split('\n')
        self.processBlocks(lines)

        for block in self.blocks:
            block.compile()

    def processBlocks(self, lines, depth=0):
        currentBlock = []
        i = 0
        while i < len(lines):
            line = lines[i]
            strippedLine = line.strip()

            if strippedLine.startswith(f'{KeywordForStart} '):
                if currentBlock:
                    self.addLlmBlock(currentBlock)
                    currentBlock = []
                
                forBlock, i = self.processForBlock(lines, i, depth + 1)
                self.blocks.append(forBlock)
            else:
                currentBlock.append(line)
            
            i += 1

        if currentBlock:
            self.addLlmBlock(currentBlock)

    def processForBlock(self, lines, startIndex, depth):
        forLines = []
        retStorage = None
        i = startIndex

        while i < len(lines):
            line = lines[i]
            strippedLine = line.strip()

            if strippedLine.startswith(KeywordForEnd):
                retStorage = self.processForEnd(strippedLine)
                break
            else:
                forLines.append(line)
            
            i += 1

        if i == len(lines) and not strippedLine.startswith(KeywordForEnd):
            # 处理未闭合的 for 循环
            forLines.append(f"{KeywordForEnd}")  # 自动添加结束标记

        return ForBlock.create(
            '\n'.join(forLines), 
            context=self.context, 
            config=self.config, 
            retStorage=retStorage,
            toolkit=self.toolkit,
            usePrevResult=self.usePrevResult
        ), i

    def processForEnd(self, line):
        parts = line.split('->')
        if len(parts) > 1:
            endPart = parts[0].strip()
            if endPart != KeywordForEnd:
                raise SyntaxError(f"Invalid for loop end statement: '{line}'. Should be 'end' or 'end -> variable_name'.")

            variable = parts[1].strip()
            if not variable.isidentifier():
                raise SyntaxError(f"Invalid variable name: '{variable}'. 'end -> ' should be followed by a valid variable name.")

            return variable
        elif line != KeywordForEnd:
            raise SyntaxError(f"Invalid for loop end statement: '{line}'. Should be 'end' or 'end -> variable_name'.")
        
        return None

    def addLlmBlock(self, blockLines):
        self.blocks.append(
            LLMBlock.create(
                context=self.context, 
                config=self.config, 
                taskExpr='\n'.join(blockLines),
                toolkit=self.toolkit,
                usePrevResult=self.usePrevResult,
                repoFuncs=self.repoFuncs
            )
        )

    def execute(
            self, 
            query: str = None, 
            args: dict = {}, 
            prevBlock: BaseBlock = None) -> Response:
        self.updateFromPrevBlock(prevBlock, args)

        result = Response()
        lastBlock = prevBlock
        for block in self.blocks:
            result = block.execute(
                query=query, 
                args=args,
                prevBlock=lastBlock)
            lastBlock = block
        return result

    @staticmethod
    def create(
            flowCode: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            repoFuncs=None) -> 'FlowBlock':
        return FlowBlock(flowCode, context, config, toolkit, usePrevResult, repoFuncs)

if __name__ == "__main__":
    flowCode = """
    Dog. #CODE 生成两个随机数 => [第个随机数，第二个随机数] -> random_nums
    for item in random_nums:
        Cat. #IF 如果{item}是奇数，返回Tiger，如果是偶数，返回Monkey
        Tiger、 根据{item}讲个笑话, #RET -> Laugh
        Monkey、 说一首诗 -> Poetry
    end
    Slime. 综合输出{Laugh}和{Poetry}
    """
    flowBlock = FlowBlock(flowCode=flowCode)
    flowBlock.compile()
    print(flowBlock.execute().response.response)
