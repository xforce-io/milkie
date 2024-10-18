from milkie.agent.agent import Agent, FakeAgentStdin
from milkie.agent.agents.base_agent import BaseAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.env.agent_program import AgentProgram
from milkie.response import Response

class Env(BaseAgent):
    def __init__(
        self, 
        context: Context = None, 
        config: str | GlobalConfig = None,
        agentPrograms: list[AgentProgram] = None
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
        for agentProgram in agentPrograms:
            self.agents[agentProgram.name] = Agent(
                code=agentProgram.getCode(), 
                context=self.context,
                config=self.config,
                toolkit=agentProgram.toolkit, 
                usePrevResult=False,
                systemPrompt=agentProgram.getSystemPrompt())
            self.agents[agentProgram.name].compile()
        
    def execute(
            self, 
            agentName: str, 
            query: str=None, 
            args: dict={}) -> Response:
        if agentName not in self.agents:
            raise RuntimeError(f"Agent[{agentName}] not found")

        return self.agents[agentName].execute(query=query, args=args)
