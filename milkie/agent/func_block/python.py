import ast
from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNodeLLM
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.utils.data_utils import preprocessPyCode, restoreVariablesInStr
from milkie.vm.vm import VM

class Python(FuncBlock):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            agentName="Python",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs)

        self.funcName = "Python"
        self.params = ["code"]

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        code = args["code"]
        formattedCode = restoreVariablesInStr(
            code, 
            self.context.varDict.getGlobalDict())
        result = self.context.vm.execPython(
            code=f"print({preprocessPyCode(formattedCode)})",
            varDict=self.context.varDict.getAllDict(),
            **kwargs)
        kwargs["execNode"].castTo(ExecNodeLLM).addContent(str(result))
        result = VM.deserializePythonResult(result)
        return Response.buildFrom(result if result else "")
        
    def createFuncCall(self):
        newFuncCall = Python(
            globalContext=self.globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs
        )
        return newFuncCall
        