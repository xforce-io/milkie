from milkie.agent.agent import Agent, FakeAgentStdin
from milkie.chatroom.chatroom import Chatroom
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.datasource import DataSource
from milkie.runtime.global_skills import GlobalSkills
from milkie.response import Response
from milkie.trace import stdout

class Env:
    def __init__(
        self, 
        context: Context = None, 
        config: str | GlobalConfig = None,
        agentPrograms: list[AgentProgram] = None,
        chatroomPrograms: list[ChatroomProgram] = None,
        globalSkillset: Skillset = None
    ) -> None:
        self.context = context
        self.config = config
        self.context.getGlobalContext().setEnv(self)
        self.dataSource = DataSource(self.context.globalContext.globalConfig)

        self.agents: dict[str, Agent] = {
            "stdin": FakeAgentStdin(
                code="", 
                context=self.context, 
                config=self.config
            )
        }
        self.chatrooms: dict[str, Chatroom] = {}

        self.globalSkillset = globalSkillset

        for agentProgram in agentPrograms:
            self.agents[agentProgram.name] = Agent(
                agentName=agentProgram.name,
                desc=agentProgram.desc,
                code=agentProgram.getCode(), 
                context=self.context.copy(),
                config=self.config,
                toolkit=agentProgram.toolkit, 
                usePrevResult=False,
                systemPrompt=agentProgram.getSystemPrompt())
        
        for chatroomProgram in chatroomPrograms:
            self.chatrooms[chatroomProgram.name] = Chatroom(
                name=chatroomProgram.name,
                desc=chatroomProgram.desc,
                host=chatroomProgram.host,
                context=self.context.copy(),
                config=self.config
            )
        
        for agent in self.agents.values():
            self.globalSkillset.addSkill(AgentToolkit(agent))

        for agentProgram in agentPrograms:
            roleAssignments = agentProgram.getRoleAssignments()
            if roleAssignments:
                for skillName, _ in roleAssignments:
                    if skillName not in self.agents:
                        self.agents[agentProgram.name].assignSkill(self.globalSkillset.getSkill(skillName))
                    else:
                        self.agents[agentProgram.name].assignSkill(AgentToolkit(self.agents[skillName]))

        for chatroomProgram in chatroomPrograms:
            # set host
            self.chatrooms[chatroomProgram.name].setHost(self.agents)

            # set roles
            roleAssignments = chatroomProgram.getRoleAssignments()
            if roleAssignments:
                for agentName, roleName in roleAssignments:
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

    def getGlobalSkillset(self) -> Skillset:
        return self.globalSkillset

    def getDataSource(self) -> DataSource:
        return self.dataSource