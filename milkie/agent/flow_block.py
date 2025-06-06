from typing import List, Union
from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNode
from milkie.agent.for_block import ForBlock
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.config.constant import DefaultUsePrevResult, KeywordForEnd, KeywordForStart
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.utils.data_utils import codeToLines

class FlowBlock(BaseBlock):
    def __init__(
            self, 
            agentName: str,
            flowCode: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None):
        super().__init__(
            agentName=agentName, 
            globalContext=globalContext, 
            config=config, 
            toolkit=toolkit, 
            repoFuncs=repoFuncs
        )
        self.flowCode = flowCode
        self.blocks: List[Union[LLMBlock, ForBlock]] = []

    def compile(self):
        if self.isCompiled:
            return

        lines = codeToLines(self.flowCode)
        self.processBlocks(lines)

        for block in self.blocks:
            block.compile()

        self.isCompiled = True

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
            agentName=self.agentName,
            forStatement='\n'.join(forLines), 
            globalContext=self.globalContext, 
            config=self.config, 
            retStorage=retStorage,
            toolkit=self.toolkit,
            repoFuncs=self.repoFuncs
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
                agentName=self.agentName,
                globalContext=self.globalContext, 
                config=self.config, 
                taskExpr='\n'.join(blockLines),
                toolkit=self.toolkit,
                repoFuncs=self.repoFuncs
            )
        )

    def execute(
            self, 
            context: Context,
            query: str,
            args: dict = {}, 
            prevBlock: BaseBlock = None,
            execNodeParent: ExecNode = None,
            **kwargs) -> Response:
        super().execute(
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock, 
            execNodeParent=execNodeParent, 
            **kwargs)
        
        result = None
        lastBlock = prevBlock
        for block in self.blocks:
            result = block.execute(
                context=context,
                query=query,
                args=args,
                prevBlock=lastBlock,
                execNodeParent=execNodeParent,
                **kwargs
            )
            lastBlock = block
        return result

    @staticmethod
    def create(
            agentName: str,
            flowCode: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            repoFuncs=None) -> 'FlowBlock':
        return FlowBlock(
            agentName=agentName,
            flowCode=flowCode,
            globalContext=globalContext,
            config=config,
            toolkit=toolkit,
            repoFuncs=repoFuncs
        )

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
