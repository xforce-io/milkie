from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.agent.step_llm_streaming import SkillTag, callSkill
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

    def execute(
            self, 
            context: Context, 
            query: str, 
            args: dict, 
            **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)

        skillName :str = args["skill"]
        toolName :str = None
        if skillName.find(".") != -1:
            skillName, toolName = skillName.split(".")
            skillName = skillName.strip()
            toolName = toolName.strip()
            if toolName == "" or skillName == "":
                raise ValueError(f"Invalid skill name[{skillName}] tool name[{toolName}]")

        skillTag = SkillTag(
            start=0,
            end=len(args["skill"]),
            skillName=skillName,
            query=query,
            toolkit=context.getGlobalContext().getEnv().getGlobalSkillset().getSkill(skillName),
            toolName=toolName
        )

        resp = callSkill(
            context=context,
            llm=kwargs["curInstruction"].getCurLLM(),
            funcCall=f"{args['skill']}({args['args']})",
            query=args['args'],
            preContext=context.history.getRecentUserPrompt(),
            skillTag=skillTag,
            **kwargs)
        return Response(respStr=resp)

    def createFuncCall(self):
        newFuncCall = SkillFunc(
            globalContext=self.globalContext, 
            config=self.config, 
            repoFuncs=self.repoFuncs
        )
        return newFuncCall