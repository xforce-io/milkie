from __future__ import annotations
from sys import stdin
from typing import List
from milkie.agent.base_block import BaseBlock
from milkie.agent.flow_block import FlowBlock
from milkie.agent.func_block.code_and_run import CodeAndRun
from milkie.agent.func_block.func_block import FuncBlock, RepoFuncs
from milkie.agent.func_block.import_func import ImportFunc
from milkie.agent.func_block.no_cache import NoCache
from milkie.agent.func_block.python import Python
from milkie.agent.func_block.read import Read
from milkie.agent.func_block.reindex_from_local_block import ReindexFromLocalBlock
from milkie.agent.func_block.retrieval_block import RetrievalBlock
from milkie.agent.func_block.set_model import SetModel
from milkie.agent.func_block.set_reasoning_self_consistency import SetReasoningSelfConsistency
from milkie.agent.func_block.set_reasoning_self_critique import SetReasoningSelfCritique
from milkie.agent.func_block.skill_func import SkillFunc
from milkie.agent.func_block.write import Write
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.config.constant import KeywordFuncStart, KeywordFuncEnd
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.agent.exec_graph import ExecNodeAgent, ExecNodeLabel, ExecNodeType
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from milkie.response import Response
import logging

from milkie.utils.data_utils import codeToLines

logger = logging.getLogger(__name__)

class Agent(BaseBlock):
    def __init__(
            self, 
            agentName: str,
            desc: str,
            code: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None,
            usePrevResult=False,
            systemPrompt: str = None):

        self.repoFuncs = RepoFuncs()

        super().__init__(
            agentName=agentName, 
            globalContext=globalContext, 
            config=config, 
            toolkit=toolkit, 
            usePrevResult=usePrevResult, 
            repoFuncs=self.repoFuncs)

        self.name = agentName
        self.desc = desc
        self.skills = dict[str, Toolkit]()
        self.code = code
        self.systemPrompt = systemPrompt
        self.funcBlocks: List[FuncBlock] = []
        self.topBlocks: List[FlowBlock] = []
        self.repoFuncs.add("Retrieval", RetrievalBlock(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("ReindexFromLocal", ReindexFromLocalBlock(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("NoCache", NoCache(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("LLM", SetModel(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("Import", ImportFunc(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("ReasoningSelfConsistency", SetReasoningSelfConsistency(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("ReasoningSelfCritique", SetReasoningSelfCritique(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("Skill", SkillFunc(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("Write", Write(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("Read", Read(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("Python", Python(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))
        self.repoFuncs.add("CodeAndRun", CodeAndRun(
            globalContext=globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        ))

    def assignSkill(self, skill: Toolkit):
        self.skills[skill.getName()] = skill

    def setCodeAndCompile(self, code: str):
        self.isCompiled = False
        self.code = code
        self.compileAsSingleLLMBlock()

    def compile(self):
        if self.isCompiled:
            return
        
        lines = codeToLines(self.code)
        currentBlock = []
        inFuncBlock = False
        funcBlockCount = 0
        for line in lines:
            strippedLine = line.strip()
            if len(strippedLine) == 0:
                continue

            if strippedLine.startswith(KeywordFuncStart):
                if currentBlock:
                    self._addFlowBlock(currentBlock)
                    currentBlock = []
                inFuncBlock = True
                currentBlock = [line]
                funcBlockCount += 1
                logger.debug(f"Found function definition: {strippedLine}")
            elif strippedLine == KeywordFuncEnd and inFuncBlock:
                currentBlock.append(line)
                self._addFuncBlock(currentBlock)
                currentBlock = []
                inFuncBlock = False
            else:
                currentBlock.append(line)

        if currentBlock:
            self._addFlowBlock(currentBlock)

        for block in self.funcBlocks:
            block.compile()

        for funcBlock in self.funcBlocks:
            funcName = funcBlock.funcName
            self.repoFuncs.add(funcName, funcBlock)
            logger.debug(f"Added function block: {funcName}")

        for block in self.topBlocks:
            block.compile()

        self.isCompiled = True

    def compileAsSingleLLMBlock(self):
        if self.isCompiled:
            return
        
        self.topBlocks = [LLMBlock.create(
            agentName=self.name,
            globalContext=self.globalContext,
            config=self.config,
            taskExpr=self.code,
            toolkit=self.toolkit,
            usePrevResult=self.usePrevResult,
            decomposeTask=False,
            repoFuncs=self.repoFuncs
        )]
        self.topBlocks[0].compile()

        self.isCompiled = True

    def _addFuncBlock(self, lines):
        funcBlock = FuncBlock.create(
            agentName=self.name,
            funcDefinition='\n'.join(lines),
            globalContext=self.globalContext,
            config=self.config,
            toolkit=self.toolkit,
            repoFuncs=self.repoFuncs
        )
        self.funcBlocks.append(funcBlock)

    def _addFlowBlock(self, lines):
        self.topBlocks.append(FlowBlock.create(
            agentName=self.name,
            flowCode='\n'.join(lines),
            globalContext=self.globalContext,
            config=self.config,
            toolkit=self.toolkit,
            repoFuncs=self.repoFuncs
        ))

    def execute(
            self, 
            context: Context = None,
            query: str = None,
            args: dict = {}, 
            prevBlock: BaseBlock = None,
            **kwargs) -> Response:
        super().execute(
            context=context,
            query=query,
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)
        
        result = None
        lastBlock = prevBlock

        if ("system_prompt" not in args or \
                args["system_prompt"] is None) and \
                self.systemPrompt:
            args["system_prompt"] = self.systemPrompt

        if len(self.skills) > 0:
            kwargs["skills"] = self.skills

        if "top" in kwargs and not kwargs["top"]:
            kwargs["history"] = None
        else:
            history = self.context.getHistory()
            history.resetUse()
            kwargs["history"] = history

        execNodeAgent = kwargs["execNode"].castTo(ExecNodeAgent)
        for block in self.topBlocks:
            result = block.execute(
                context=context,
                query=query,
                args=args,
                prevBlock=lastBlock,
                execNodeParent=execNodeAgent,
                **{k: v for k, v in kwargs.items() if k != "top"}
            )
            lastBlock = block

        self.context.historyAddAssistantPrompt(result.respStr)
        return result

class FakeAgentStdin(Agent):
    def __init__(
            self, 
            code: str, 
            globalContext: GlobalContext = None, 
            config: str | GlobalConfig = None, 
            toolkit: Toolkit = None, 
            usePrevResult=False, 
            systemPrompt: str = None):
        super().__init__(
            agentName="fake stdin",
            desc="mock stdin agent",
            code=code,
            globalContext=globalContext,
            config=config,
            toolkit=toolkit,
            usePrevResult=usePrevResult,
            systemPrompt=systemPrompt
        )

    def execute(
            self, 
            context: Context,
            query: str = None,
            args: dict = {}, 
            prevBlock: BaseBlock = None,
            **kwargs) -> Response:
        resp = stdin.readline()
        return Response(respStr=resp)

if __name__ == "__main__":
    code = """
    DEF down(ceiling)
        返回比{ceiling}小的最大的奇数，直接返回结果
    END

    1. 10 以内最大的质数 -> num
    2. 以@down({num})为主题写一首诗
    """
    agent = Agent(
        agentName="test", 
        desc="test agent", 
        code=code)
    agent.compile()
    print(agent.execute().resp)
