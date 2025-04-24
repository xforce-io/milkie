import os
from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response


class Write(FuncBlock):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="Write",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs)
        
        self.funcName = "Write"
        self.params = ["filepath", "varname"]

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)
        
        self._restoreParams(args, self.params)
        filepath = args["filepath"]
        varname = args["varname"]
        varDict = context.varDict.getAllDict()
        if varname not in varDict:
            raise ValueError(f"varname {varname} not found in varDict")
        content = varDict[varname]

        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(filepath, "w") as f:
            f.write(content)
        return Response(respStr=f"write to {filepath} success")
    
    def createFuncCall(self):
        newFuncCall = Write(
            globalContext=self.globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        )
        return newFuncCall