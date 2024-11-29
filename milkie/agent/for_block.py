import json
import logging
import re
from milkie.agent.base_block import BaseBlock
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult, KeywordForStart, KeyRet
from milkie.context import Context, VarDict
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.response import Response

logger = logging.getLogger(__name__)

class ForBlock(BaseBlock):
    def __init__(
            self, 
            forStatement: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
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
            toolkit=self.toolkit,
            taskExpr=self.loopBody,
            decomposeTask=True,
            repoFuncs=self.repoFuncs
        )
        self.loopBlock.compile()

    def validate(self, args: VarDict):
        variable = ""
        rest = None
        for key, _ in args.getAllDict().items():
            if self.iterable.startswith(key) and len(key) > len(variable):
                variable = key
                rest = self.iterable[len(key):]

        try:
            iterableValue = eval(f"args.get('{variable}'){rest}")
        except Exception as e:
            raise ValueError(f"Iterable '{self.iterable}' not found in variable dictionary")

        if isinstance(iterableValue, dict):
            self.loopType = dict
        elif isinstance(iterableValue, (list, tuple)):
            self.loopType = list
        elif isinstance(iterableValue, str):
            try:
                iterableValue = iterableValue.replace("'", '"')
                iterableValue = json.loads(iterableValue)
                if isinstance(iterableValue, dict):
                    self.loopType = dict
                elif isinstance(iterableValue, (list, tuple)):
                    self.loopType = list
                else:
                    raise ValueError(f"Iterable '{self.iterable}' must be a list, tuple, or dict")
            except json.JSONDecodeError:
                import ast
                try:
                    iterableValue = ast.literal_eval(iterableValue)
                    if isinstance(iterableValue, dict):
                        self.loopType = dict
                    elif isinstance(iterableValue, (list, tuple)):
                        self.loopType = list
                    else:
                        raise ValueError(f"Iterable '{self.iterable}' must be a list, tuple, or dict")
                except:
                    raise ValueError(f"Cannot parse string value as iterable: {iterableValue}")
        else:
            raise ValueError(f"Iterable '{self.iterable}' must be a list, tuple, or dict")
        return iterableValue

    def execute(
            self, 
            context: Context,
            query: str = None, 
            args: dict = {}, 
            prevBlock :BaseBlock=None,
            **kwargs) -> Response:
        super().execute(
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock,
            **kwargs)

        iterableValue = self.validate(self.getVarDict())
        results = []
        if self.loopType == dict:
            items = iterableValue.items()
        else:
            items = enumerate(iterableValue)

        for key, value in items:
            if self.loopType == dict:
                self.setVarDictGlobal(
                    self.loopVar, 
                    { "key": key, "value": value})
            else:
                self.setVarDictGlobal(self.loopVar, value)

            try:
                result = self.loopBlock.execute(
                    context=context,
                    query=query, 
                    args=args,
                    prevBlock=prevBlock,
                    **kwargs)
                if result.resp != KeyRet:
                    results.append(result.resp)
            except Exception as e:
                logger.warning(f"Error in loop block: {e}")
            prevBlock = None

        if self.retStorage:
            self.setVarDictGlobal(self.retStorage, results)
        return Response(respList=results)

    def __str__(self):
        return f"ForBlock(loopVar={self.loopVar}, iterable={self.iterable}, loopType={self.loopType.__name__ if self.loopType else 'None'})"

    @staticmethod
    def create(
            forStatement: str, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            loopBlockClass=LLMBlock,
            retStorage=None,
            repoFuncs=None) -> 'ForBlock':
        return ForBlock(
            forStatement=forStatement,
            context=context,
            config=config,
            toolkit=toolkit,
            usePrevResult=usePrevResult,
            loopBlockClass=loopBlockClass,
            retStorage=retStorage,
            repoFuncs=repoFuncs
        )
