from typing import List, Union
from milkie.agent.base_block import BaseBlock
from milkie.agent.llm_block import LLMBlock, Response
from milkie.agent.for_block import ForBlock
from milkie.config.constant import DefaultUsePrevResult, KeywordForEnd, KeywordForStart
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.functions.toolkits.base_toolkits import BaseToolkit

class FlowBlock(BaseBlock):
    def __init__(
            self, 
            flowCode: str, 
            context :Context=None, 
            config :str|GlobalConfig=None,
            toolkit :BaseToolkit=None,
            usePrevResult=DefaultUsePrevResult):
        super().__init__(context, config, toolkit, usePrevResult)

        self.flowCode = flowCode
        self.blocks: List[Union[LLMBlock, ForBlock]] = []

    def compile(self):
        lines = self.flowCode.split('\n')
        currentBlock = []
        inForLoop = False
        forBlock = []
        retStorage = None

        for line in lines:
            strippedLine = line.strip()
            
            if strippedLine.startswith(f'{KeywordForStart} ') and strippedLine.endswith(":"):
                if inForLoop:
                    # 如果已经在 for 循环中，结束当前的 ForBlock
                    self.blocks.append(
                        ForBlock(
                            '\n'.join(forBlock), 
                            context=self.context, 
                            config=self.config, 
                            toolkit=self.toolkit,
                            usePrevResult=self.usePrevResult))
                    forBlock = []
                elif currentBlock:
                    self.blocks.append(
                        LLMBlock(
                            context=self.context, 
                            config=self.config, 
                            taskExpr='\n'.join(currentBlock),
                            toolkit=self.toolkit,
                            usePrevResult=self.usePrevResult))
                    currentBlock = []
                
                inForLoop = True
                forBlock = [line]
            elif inForLoop:
                if strippedLine.startswith(KeywordForEnd):
                    parts = strippedLine.split('->')
                    if len(parts) > 1:
                        end_part = parts[0].strip()
                        if end_part != KeywordForEnd:
                            raise SyntaxError(f"Invalid for loop end statement: '{strippedLine}'. Should be 'end' or 'end -> variable_name'.")

                        variable = parts[1].strip()
                        if not variable.isidentifier():
                            raise SyntaxError(f"Invalid variable name: '{variable}'. 'end -> ' should be followed by a valid variable name.")

                        retStorage = variable
                    elif strippedLine != KeywordForEnd:
                        raise SyntaxError(f"Invalid for loop end statement: '{strippedLine}'. Should be 'end' or 'end -> variable_name'.")
                    
                    # 结束当前的 ForBlock
                    self.blocks.append(
                        ForBlock(
                            '\n'.join(forBlock), 
                            context=self.context, 
                            config=self.config, 
                            retStorage=retStorage,
                            toolkit=self.toolkit,
                            usePrevResult=self.usePrevResult))
                    inForLoop = False
                    forBlock = []
                else:
                    forBlock.append(line)
            else:
                currentBlock.append(line)

        # 处理最后一个块
        if inForLoop:
            self.blocks.append(
                ForBlock(
                    '\n'.join(forBlock), 
                    context=self.context, 
                    config=self.config, 
                    retStorage=retStorage,
                    toolkit=self.toolkit,
                    usePrevResult=self.usePrevResult))
        elif currentBlock:
            self.blocks.append(
                LLMBlock(
                    context=self.context, 
                    config=self.config, 
                    taskExpr='\n'.join(currentBlock),
                    toolkit=self.toolkit,
                    usePrevResult=self.usePrevResult))

        for block in self.blocks:
            block.compile()

    def execute(
            self, 
            query: str = None, 
            args: dict = {}, 
            prevBlock :BaseBlock=None) -> Response:
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
