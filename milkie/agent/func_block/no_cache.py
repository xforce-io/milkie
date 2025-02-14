from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response


class NoCache(FuncBlock):

    def __init__(
            self, 
            context: Context, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="NoCache", 
            context=context, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "NoCache"
        self.params = []

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        kwargs["curInstruction"].noCache = True
        return Response(respStr="set no cache")
