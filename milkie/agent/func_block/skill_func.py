from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response


class SkillFunc(FuncBlock):
    def __init__(self, globalContext: GlobalContext, config: str, repoFuncs=None):
        super().__init__(
            agentName="SkillFunc",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs)

        self.funcName = "Skill"
        self.params = ["skill", "args"]

    def execute(self, context: Context, args: dict, **kwargs):
        BaseBlock.execute(self, context, args, **kwargs)

        context.globalContext.getEnv().getGlobalSkillset().getSkill(args["skill"]).execute(args["args"])
        return Response(respStr="skill executed")