from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response


class NoCache(FuncBlock):

    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="NoCache", 
            globalContext=globalContext, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "NoCache"
        self.params = []

    def execute(self, context: Context, args: dict, **kwargs):
        BaseBlock.execute(self, context, args, **kwargs)

        kwargs["curInstruction"].noCache = True
        return Response(respStr="set no cache")
