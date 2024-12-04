from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.llm.reasoning.reasoning_self_consistency import ReasoningSelfConsistency
from milkie.response import Response


class SetReasoningSelfConsistency(FuncBlock):

    def __init__(
            self, 
            context: Context, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            context=context, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "ReasoningSelfConsistency"
        self.params = ["amateur", "expert"]

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        amateur = args.get("amateur")
        if not amateur:
            amateur = kwargs["curInstruction"].llm
        else:
            amateur = self.context.globalContext.settings.getLLM(amateur)

        if not amateur:
            raise ValueError(f"LLM {amateur} not found")
        
        kwargs["curInstruction"].reasoning = ReasoningSelfConsistency(
            self.context.globalContext,
            amateur)
        return Response(respStr="set reasoning self consistency")
