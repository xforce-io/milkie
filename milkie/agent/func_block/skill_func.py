import ast
import json
from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.agent.step_llm_streaming import SkillTag, callSkill
from milkie.context import Context
from milkie.functions.openai_function import OpenAIFunction
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
        self.globalSkillset = globalContext.getEnv().getGlobalSkillset()

    def execute(
            self, 
            context: Context, 
            query: str, 
            args: dict, 
            **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)

        skillName :str = args["skill"]
        args = args["args"]

        skillAndToolName = skillName.split("::")
        if len(skillAndToolName) == 2:
            skillName = skillAndToolName[0]
            toolName = skillAndToolName[1]
            tool :OpenAIFunction = self.globalSkillset.getTool(skillName, toolName)
            parameters = tool.openai_tool_schema["function"]["parameters"]["properties"]
            if tool:
                if len(parameters) == 1:
                    result = None
                    if list(parameters.values())[0]["type"] == "array":
                        result = tool.func(ast.literal_eval(args))
                    else:
                        result = tool.func(args)
                    context.genResp(result)
                    return Response.buildFrom(result)
                else:
                    raise Exception("TO BE implemented")
        
        toolName :str = None
        if skillName.find(".") != -1:
            skillName, toolName = skillName.split(".")
            skillName = skillName.strip()
            toolName = toolName.strip()
            if toolName == "" or skillName == "":
                raise ValueError(f"Invalid skill name[{skillName}] tool name[{toolName}]")

        skillTag = SkillTag(
            start=0,
            end=len(skillName),
            skillName=skillName,
            query=query,
            toolkit=context.getGlobalContext().getEnv().getGlobalSkillset().getSkill(skillName),
            toolName=toolName
        )

        #if "no_cache" in args and args["no_cache"]:
        #    kwargs["no_cache"] = True

        resp = callSkill(
            context=context,
            llm=kwargs["curInstruction"].getCurLLM(),
            funcCall=f"{skillName}({args})",
            query=args,
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