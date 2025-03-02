from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response


class ImportFunc(FuncBlock):
    def __init__(
            self, 
            context: Context, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="ImportFunc", 
            context=context, 
            config=config, 
            repoFuncs=repoFuncs
        )

        self.funcName = "Import"
        self.params = ["toolkit"]

    def execute(
            self, 
            context: Context, 
            query: str, 
            args: dict, 
            **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        toolkit = args["toolkit"]
        kwargs["curInstruction"].toolkit = context.globalContext.getEnv().getGlobalToolkits().getToolkit(toolkit)
        return Response(respStr="imported toolkit")
