from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response
from milkie.global_context import GlobalContext

class ImportFunc(FuncBlock):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="ImportFunc", 
            globalContext=globalContext, 
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
        kwargs["curInstruction"].toolkit = context.globalContext.getEnv().getGlobalSkillset().getSkill(toolkit)
        return Response(respStr="imported toolkit")

