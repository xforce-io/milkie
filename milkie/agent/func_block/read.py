from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response


class Read(FuncBlock):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="Read",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs
        )

        self.funcName = "Read"
        self.params = ["filepath"]

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        filepath = args["filepath"]
        with open(filepath, "r") as f:
            content = f.read()
        return Response(respStr=content)
    
    def createFuncCall(self):
        newFuncCall = Read(
            globalContext=self.globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        )
        return newFuncCall