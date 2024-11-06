from milkie.agent.agent import Agent, FakeAgentStdin
from milkie.agent.agents.base_agent import BaseAgent
from milkie.chatroom.chatroom import Chatroom
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.global_toolkits import GlobalToolkits
from milkie.response import Response
from milkie.trace import stdout

class Env:
    def __init__(
        self, 
        context: Context = None, 
        config: str | GlobalConfig = None,
        agentPrograms: list[AgentProgram] = None,
        chatroomPrograms: list[ChatroomProgram] = None,
        globalToolkits: GlobalToolkits = None
    ) -> None:
        self.context = context
        self.config = config
        self.context.getGlobalContext().setEnv(self)

        self.agents: dict[str, Agent] = {
            "stdin": FakeAgentStdin(
                code="", 
                context=self.context, 
                config=self.config
            )
        }
        self.chatrooms: dict[str, Chatroom] = {}

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
        
        for chatroomProgram in chatroomPrograms:
            self.chatrooms[chatroomProgram.name] = Chatroom(
                name=chatroomProgram.name,
                desc=chatroomProgram.desc,
                host=chatroomProgram.host,
                context=self.context,
                config=self.config
            )
        
        for agent in self.agents.values():
            self.globalToolkits.addAgent(agent)

        for agentProgram in agentPrograms:
            expertAssignments = agentProgram.getExpertAssignments()
            if expertAssignments:
                for agentName, _ in expertAssignments:
                    self.agents[agentProgram.name].assignExpert(self.agents[agentName])

        for chatroomProgram in chatroomPrograms:
            # set host
            self.chatrooms[chatroomProgram.name].setHost(self.agents)

            # set roles
            expertAssignments = chatroomProgram.getExpertAssignments()
            if expertAssignments:
                for agentName, roleName in expertAssignments:
                    self.chatrooms[chatroomProgram.name].assignRole(self.agents[agentName], roleName)

            # set prologue
            self.chatrooms[chatroomProgram.name].prologue = chatroomProgram.prologue

        for agent in self.agents.values():
            agent.compile()

        for chatroom in self.chatrooms.values():
            chatroom.compile()
        
    def execute(
            self, 
            chatroomName: str=None,
            agentName: str=None, 
            query: str=None, 
            args: dict={},
            **kwargs) -> Response:
        if chatroomName and chatroomName not in self.chatrooms:
            raise RuntimeError(f"Chatroom[{chatroomName}] not found")
        elif agentName and agentName not in self.agents:
            raise RuntimeError(f"Agent[{agentName}] not found")

        if chatroomName:
            stdout(f"\n <<< start of chatroom[{chatroomName}] with query {query} >>> ", **kwargs)
            response = self.chatrooms[chatroomName].execute(query=query, args=args, **kwargs)
            stdout(f"\n <<< end of chatroom[{chatroomName}] >>>\n", **kwargs)
            return response
        elif agentName:
            stdout(f"\n <<< start of agent[{agentName}] with query {query} >>> ", **kwargs)
            response = self.agents[agentName].execute(query=query, args=args, top=True, **kwargs)
            stdout(f"\n <<< end of agent[{agentName}] >>>\n", **kwargs)
            return response

    def getGlobalToolkits(self) -> GlobalToolkits:
        return self.globalToolkits