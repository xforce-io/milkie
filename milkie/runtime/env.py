from milkie.agent.agent import Agent, FakeAgentStdin
from milkie.agent.agents.base_agent import BaseAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.global_toolkits import GlobalToolkits
from milkie.response import Response
from milkie.trace import stdout

class Env(BaseAgent):
    def __init__(
        self, 
        context: Context = None, 
        config: str | GlobalConfig = None,
        agentPrograms: list[AgentProgram] = None,
        globalToolkits: GlobalToolkits = None
    ) -> None:
        super().__init__(context, config)

        self.context.getGlobalContext().setEnv(self)

        self.agents = {
            "stdin": FakeAgentStdin(
                code="", 
                context=self.context, 
                config=self.config
            )
        }

        self.globalToolkits = globalToolkits

        for agentProgram in agentPrograms:
            self.agents[agentProgram.name] = Agent(
                name=agentProgram.name,
                desc=agentProgram.desc,
                code=agentProgram.getCode(), 
                context=self.context,
                config=self.config,
                toolkit=agentProgram.toolkit, 
                usePrevResult=False,
                systemPrompt=agentProgram.getSystemPrompt())
        
        for agent in self.agents.values():
            self.globalToolkits.addAgent(agent)

        for agent in self.agents.values():
            agent.compile()
        
    def execute(
            self, 
            agentName: str, 
            query: str=None, 
            args: dict={},
            **kwargs) -> Response:
        if agentName not in self.agents:
            raise RuntimeError(f"Agent[{agentName}] not found")

        stdout(f"\n <<< start of agent[{agentName}] with query {query} >>> ", args=args, **kwargs)
        response = self.agents[agentName].execute(query=query, args=args, **kwargs)
        stdout(f"\n <<< end of agent[{agentName}] >>>\n", args=args, **kwargs)
        return response

    def getGlobalToolkits(self) -> GlobalToolkits:
        return self.globalToolkits