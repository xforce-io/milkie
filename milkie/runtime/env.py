from typing import Optional
from milkie.agent.agent import Agent, FakeAgentStdin
from milkie.agent.exec_graph import ExecNodeAgent, ExecNodeLabel, ExecNodeRoot, ExecNodeSkill, ExecNodeType
from milkie.chatroom.chatroom import Chatroom
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.global_context import GlobalContext
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.datasource import DataSource
from milkie.response import Response
from milkie.types.object_type import ObjectTypeFactory

class Env:
    def __init__(
        self, 
        globalContext: GlobalContext = None, 
        config: str | GlobalConfig = None,
        agentPrograms: list[AgentProgram] = None,
        chatroomPrograms: list[ChatroomProgram] = None,
        globalSkillset: Skillset = None,
        globalObjectTypes: ObjectTypeFactory = None
    ) -> None:
        self.globalContext = globalContext
        self.config = config
        self.globalContext.setEnv(self)
        self.dataSource = DataSource(self.globalContext.globalConfig)

        self.agents: dict[str, Agent] = {
            "stdin": FakeAgentStdin(
                code="", 
                globalContext=self.globalContext, 
                config=self.config
            )
        }
        self.chatrooms: dict[str, Chatroom] = {}

        self.globalSkillset = globalSkillset
        self.globalObjectTypes = globalObjectTypes

        for agentProgram in agentPrograms:
            self.agents[agentProgram.name] = Agent(
                agentName=agentProgram.name,
                desc=agentProgram.desc,
                code=agentProgram.getCode(), 
                globalContext=self.globalContext,
                config=self.config,
                toolkit=agentProgram.toolkit, 
                usePrevResult=False,
                systemPrompt=agentProgram.getSystemPrompt())

        for chatroomProgram in chatroomPrograms:
            self.chatrooms[chatroomProgram.name] = Chatroom(
                name=chatroomProgram.name,
                desc=chatroomProgram.desc,
                host=chatroomProgram.host,
                globalContext=self.globalContext,
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

    def getAllAgents(self) -> dict[str, Agent]:
        return self.agents

    def getAgent(self, name: str) -> Optional[Agent]:
        return self.agents.get(name)

    def getAllChatrooms(self) -> dict[str, Chatroom]:
        return self.chatrooms

    def getChatroom(self, name: str) -> Optional[Chatroom]:
        return self.chatrooms.get(name)

    def execute(
            self, 
            context: Context,
            query: str,
            chatroomName: str=None,
            agentName: str=None, 
            args: dict={},
            **kwargs) -> Response:
        if chatroomName and chatroomName not in self.chatrooms:
            raise RuntimeError(f"Chatroom[{chatroomName}] not found")
        elif agentName and agentName not in self.agents:
            raise RuntimeError(f"Agent[{agentName}] not found")

        if chatroomName:
            context.genResp(f"\n <<< start of chatroom[{chatroomName}] with query {query} >>> ", **kwargs)
            response = self.chatrooms[chatroomName].execute(
                context=context,
                query=query,
                args=args, 
                **kwargs)
            context.genResp(f"\n <<< end of chatroom[{chatroomName}] >>>\n", **kwargs)
            return response
        elif agentName:
            context.genResp(f"\n <<< start of agent[{agentName}] with query {query} >>> ", **kwargs)

            if kwargs["execNode"].label == ExecNodeLabel.ROOT:
                execNodeRoot :ExecNodeRoot = kwargs["execNode"]
                execNodeAgent = ExecNodeAgent.build(
                    execGraph=context.getExecGraph(),
                    callee=execNodeRoot,
                    name=agentName)
            elif kwargs["execNode"].label == ExecNodeLabel.SKILL:
                execNodeSkill :ExecNodeSkill = kwargs["execNode"]
                execNodeAgent = ExecNodeAgent.build(
                    execGraph=context.getExecGraph(),
                    callee=execNodeSkill,
                    name=agentName)
            else:
                raise RuntimeError(f"Invalid execNode label[{kwargs['execNode'].label}]")

            response = self.agents[agentName].execute(
                context=context,
                query=query,
                args=args, 
                top=True, 
                **{**kwargs, "execNode" : execNodeAgent})
            context.genResp(f"\n <<< end of agent[{agentName}] >>>\n", **kwargs)
            return response

    def getGlobalSkillset(self) -> Skillset:
        return self.globalSkillset

    def getGlobalObjectTypes(self) -> ObjectTypeFactory:
        return self.globalObjectTypes

    def getDataSource(self) -> DataSource:
        return self.dataSource