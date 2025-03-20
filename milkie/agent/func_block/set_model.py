from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response
from milkie.global_context import GlobalContext

class SetModel(FuncBlock):

    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="SetModel", 
            globalContext=globalContext, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "LLM"
        self.params = ["name"]

    def execute(
            self, 
            context: Context, 
            query: str,
            args: dict, 
            **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        name = args["name"]
        llm = context.globalContext.settings.getLLM(name)
        if llm is None:
            raise RuntimeError(f"model[{name}] not found")
        
        kwargs["curInstruction"].llm = llm
        return Response(respStr="set model to " + name)

    def createFuncCall(self):
        newFuncCall = SetModel(
            globalContext=self.globalContext, 
            config=self.config, 
            repoFuncs=self.repoFuncs
        )
        return newFuncCall