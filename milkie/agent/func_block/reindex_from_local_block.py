from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response
from milkie.global_context import GlobalContext

class ReindexFromLocalBlock(FuncBlock):

    def __init__(
            self, 
            globalContext: GlobalContext, 
            config :str = None, 
            repoFuncs=None):
        super().__init__(
            agentName="ReindexFromLocal", 
            globalContext=globalContext, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "ReindexFromLocal"
        self.params = ["localDir"]

    def execute(
            self, 
            context: Context, 
            query: str,
            args: dict, 
            **kwargs) -> Response:
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        localDir = args["localDir"]
        self._rebuildFromLocalDir(localDir)
        return Response(respStr="reindex from local dir: " + localDir)

    def createFuncCall(self):
        newFuncCall = ReindexFromLocalBlock(
            globalContext=self.globalContext, 
            config=self.config, 
            repoFuncs=self.repoFuncs
        )
        return newFuncCall

    def _rebuildFromLocalDir(self, localDir :str):
        self.context.getEnv().dataSource.getMainRetriever().rebuildFromLocalDir(localDir)
