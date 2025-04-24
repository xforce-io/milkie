from milkie.agent.base_block import BaseBlock
from milkie.agent.exec_graph import ExecNodeLLM
from milkie.agent.func_block.func_block import FuncBlock
from milkie.config.constant import KeywordCurrent
from milkie.context import Context
from milkie.functions.code_interpreter import CodeInterpreter
from milkie.global_context import GlobalContext
from milkie.response import Response


class CodeAndRun(FuncBlock):
    def __init__(self, globalContext: GlobalContext, config: str, repoFuncs=None):
        super().__init__(
            agentName="CodeAndRun",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs)

        self.funcName = "CodeAndRun"
        self.params = ["request", "condition"]

        self.codeInterpreter = CodeInterpreter(self.globalContext)
        self.numAttempts = 3

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)

        request = args["request"]
        condition = args["condition"]
        result = ""
        for i in range(self.numAttempts):
            result = self.codeInterpreter.execute(
                instruction=request,
                varDict=self.context.varDict.getAllDict(),
                vm=self.context.vm,
                **kwargs)
            if result == None or not Response.isNaivePyType(result):
                result = ""

            if condition:
                self.context.varDict.setLocal(KeywordCurrent, result)
                result = self.context.vm.execPython(
                    code=f"print({condition})",
                    varDict=self.context.varDict.getAllDict())
                if result == "True":
                    break

        self._getContext().genResp(result, **kwargs)
        kwargs["execNode"].castTo(ExecNodeLLM).addContent(str(result))
        return Response.buildFrom(result if result else "")

    def createFuncCall(self):
        newFuncCall = CodeAndRun(
            globalContext=self.globalContext, 
            config=self.config, 
            repoFuncs=self.repoFuncs
        )
        return newFuncCall