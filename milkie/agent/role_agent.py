from llama_index.core import Response
from milkie.agent.base_agent import BaseAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.llm.inference import chat
from milkie.tools.tool import Coding


class RoleAgent(BaseAgent):
    
    def __init__(
            self, 
            context: Context, 
            config: str,
            role: str,
            goal: str,
            backstory: str,
            tools :list,
            plan :str) -> None:
        super().__init__(context, config)

        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.plan = plan
    
    def execute(self, query: str, **kwargs) -> Response:
        self.plan = self._makePlan(query, **kwargs)
        return None

    def _makePlan(self, query: str, **kwargs) -> Response:
        if self.plan:
            return self.plan

        systemPrompt = f"""
        Role: {self.role}
        Goal: {self.goal}
        Backstory: {self.backstory}
        """

        prompt = f"""
        我们有这些工具
        Tools: {self.tools}
        Query: {query}
        现在请为 Query制定一个计划
        """

        response = chat(
            llm=self.context.globalContext.settings.llm, 
            systemPrompt=systemPrompt,
            prompt=prompt, 
            promptArgs={}, 
            **kwargs)
        return response.response

if __name__ == "__main__":
    globalConfig = GlobalConfig("config/global.yaml")
    context = Context(globalConfig)
    roleAgent = RoleAgent(
        context=context,
        config=None,
        role="你是一个仔细认真的数据分析师，需要使用数据分析工具来解决问题",
        goal="快速分析和洞察数据，提供有用的见解",
        backstory=None,
        tools=[Coding()],
        plan=None)

    response = roleAgent.execute("data路径下有个名为‘NPS分析’的文件，需要做一些洞察分析")
    print(response.plan)