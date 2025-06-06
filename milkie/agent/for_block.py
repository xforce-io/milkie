import json
import logging
import re
from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNode, ExecNodeAgent, ExecNodeFor, ExecNodeLabel, ExecNodeSequence, ExecNodeType
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.config.config import GlobalConfig
from milkie.config.constant import DefaultUsePrevResult, KeywordForStart, KeyRet
from milkie.context import Context, VarDict
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.utils.data_utils import codeToLines

logger = logging.getLogger(__name__)

class ForBlock(BaseBlock):
    def __init__(
            self, 
            agentName: str,
            forStatement: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            loopBlockClass=LLMBlock,
            retStorage=None,
            repoFuncs=None):
        super().__init__(
            agentName=agentName, 
            globalContext=globalContext, 
            config=config, 
            toolkit=toolkit, 
            usePrevResult=usePrevResult, 
            repoFuncs=repoFuncs)

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
        lines = codeToLines(self.forStatement)
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
            agentName=self.agentName,
            globalContext=self.globalContext,
            config=self.config,
            toolkit=self.toolkit,
            taskExpr=self.loopBody,
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
            args: dict = {}, 
            prevBlock :BaseBlock=None,
            execNodeParent: ExecNode = None,
            **kwargs) -> Response:
        super().execute(
            context=context, 
            args=args, 
            prevBlock=prevBlock,
            execNodeParent=execNodeParent,
            **kwargs)

        iterableValue = self.validate(self.getVarDict())
        results = []
        if self.loopType == dict:
            items = iterableValue.items()
        else:
            items = enumerate(iterableValue)

        assert execNodeParent.label == ExecNodeLabel.AGENT
        execNodeAgent :ExecNodeAgent = execNodeParent
        execNodeFor :ExecNodeFor = ExecNodeFor.build(
            execGraph=execNodeAgent.execGraph,
            execNodeAgent=execNodeAgent)

        for key, value in items:
            if self.loopType == dict:
                self.setVarDictGlobal(
                    self.loopVar, 
                    { "key": key, "value": value})
            else:
                self.setVarDictGlobal(self.loopVar, value)

            execNodeSequence :ExecNodeSequence = ExecNodeSequence.build(
                execGraph=execNodeFor.execGraph,
                context={
                    "loopVar": key,
                    "loopValue": value
                })
            execNodeFor.addExecute(execNodeSequence)

            try:
                result = self.loopBlock.execute(
                    context=context,
                    args=args,
                    prevBlock=prevBlock,
                    execNodeParent=execNodeSequence,
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
            agentName: str,
            forStatement: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=DefaultUsePrevResult,
            loopBlockClass=LLMBlock,
            retStorage=None,
            repoFuncs=None) -> 'ForBlock':
        return ForBlock(
            agentName=agentName,
            forStatement=forStatement,
            globalContext=globalContext,
            config=config,
            toolkit=toolkit,
            usePrevResult=usePrevResult,
            loopBlockClass=loopBlockClass,
            retStorage=retStorage,
            repoFuncs=repoFuncs
        )
