import re
from milkie.agent.base_block import BaseBlock
from milkie.agent.llm_block import LLMBlock, Response
from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult, KeywordForStart, KeyRet
from milkie.context import Context
from milkie.functions.toolkits.base_toolkits import BaseToolkit

class ForBlock(BaseBlock):
    def __init__(
            self, 
            forStatement: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: BaseToolkit = None,
            usePrevResult=DefaultUsePrevResult,
            loopBlockClass=LLMBlock,
            retStorage=None,
            repoFuncs=None):
        super().__init__(context, config, toolkit, usePrevResult, repoFuncs)
        self.forStatement = forStatement.strip()
        self.loopVar = None
        self.iterable = None
        self.loopBody = None
        self.loopType = None
        self.loopBlockClass = loopBlockClass
        self.retStorage = retStorage
        self.loopBlock = None
        self.parseForStatement()

    def parseForStatement(self):
        lines = self.forStatement.split('\n')
        for_line = lines[0].strip()
        
        pattern = r'%s\s+(\w+)\s+in\s+(.+)\s*' % KeywordForStart
        match = re.match(pattern, for_line)
        if not match:
            raise ValueError("Invalid for statement syntax")
        
        self.loopVar = match.group(1)
        self.iterable = match.group(2)
        self.loopBody = '\n'.join(lines[1:]).strip()

        if not self.loopBody:
            self.loopBody = "pass"

    def compile(self):
        self.loopBlock = self.loopBlockClass(
            context=self.context,
            config=self.config,
            taskExpr=self.loopBody,
            decomposeTask=True,
            repoFuncs=self.repoFuncs
        )
        self.loopBlock.compile()

    def validate(self, args: dict):
        variable = ""
        rest = None
        for key, _ in args.items():
            if self.iterable.startswith(key) and len(key) > len(variable):
                variable = key
                rest = self.iterable[len(key):]

        try:
            iterableValue = eval(f"args['{variable}']{rest}")
        except Exception as e:
            raise ValueError(f"Iterable '{self.iterable}' not found in variable dictionary")

        if isinstance(iterableValue, dict):
            self.loopType = dict
        elif isinstance(iterableValue, (list, tuple)):
            self.loopType = list
        else:
            raise ValueError(f"Iterable '{self.iterable}' must be a list, tuple, or dict")
        return iterableValue

    def execute(
            self, 
            query: str = None, 
            args: dict = {}, 
            prevBlock :BaseBlock=None) -> Response:
        self.updateFromPrevBlock(prevBlock, args)
        iterableValue = self.validate(self.getVarDict())
        results = []
        if self.loopType == dict:
            items = iterableValue.items()
        else:
            items = enumerate(iterableValue)

        for key, value in items:
            if self.loopType == dict:
                self.loopBlock.setVarDict(
                    self.loopVar, 
                    { "key": key, "value": value})
            else:
                self.loopBlock.setVarDict(self.loopVar, value)

            result = self.loopBlock.execute(
                query=query, 
                prevBlock=prevBlock)
            if result.resp != KeyRet:
               results.append(result.resp)
            prevBlock = None

        self.setVarDict(self.retStorage, results)
        return Response(respList=results)

    def __str__(self):
        return f"ForBlock(loopVar={self.loopVar}, iterable={self.iterable}, loopType={self.loopType.__name__ if self.loopType else 'None'})"

    @staticmethod
    def create(
            forStatement: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: BaseToolkit = None,
            usePrevResult=DefaultUsePrevResult,
            loopBlockClass=LLMBlock,
            retStorage=None,
            repoFuncs=None) -> 'ForBlock':
        return ForBlock(forStatement, context, config, toolkit, usePrevResult, loopBlockClass, retStorage, repoFuncs)