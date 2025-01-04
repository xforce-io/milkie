from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.llm.reasoning.reasoning_self_critique import ReasoningSelfCritique
from milkie.response import Response

class SetReasoningSelfCritique(FuncBlock):

    def __init__(
            self, 
            context: Context, 
            config: str, 
            repoFuncs=None):
        super().__init__(
            context=context, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "ReasoningSelfCritique"
        self.params = ["critique"]

    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        critiqueLLM = args.get("critique")
        if not critiqueLLM:
            critiqueLLM = kwargs["curInstruction"].llm 
        if not critiqueLLM:
            critiqueLLM = context.globalContext.settings.getLLMDefault()
        if not critiqueLLM:
            raise ValueError(f"LLM {critiqueLLM} not found")
        
        kwargs["curInstruction"].reasoning = ReasoningSelfCritique(
            globalContext=self.context.globalContext,
            llm=critiqueLLM)
        return Response(respStr="set reasoning self critique") 
