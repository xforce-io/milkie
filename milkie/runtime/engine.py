import os
from typing import Generator, Optional
from milkie.agent.agent import Agent
from milkie.chatroom.chatroom import Chatroom
from milkie.context import Context
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.env import Env
from milkie.runtime.global_skills import GlobalSkills
from milkie.global_context import GlobalContext
import logging

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
            self,
            folder: str = None,
            file: str = None,
            config: str = None) -> None:
        Context.globalContext = GlobalContext.create(config)
        self.globalSkills = GlobalSkills(Context.globalContext)
        self.globalSkillset = self.globalSkills.createSkillset()

        self.agentPrograms = []
        self.chatroomPrograms = []
        if folder:
            for filename in os.listdir(folder):
                if filename.endswith('.at'):
                    programFilepath = os.path.join(folder, filename)
                    program = AgentProgram(
                        programFilepath=programFilepath,
                        globalSkillset=self.globalSkillset,
                        globalContext=Context.globalContext
                    )
                    program.parse()
                    self.agentPrograms.append(program)
                elif filename.endswith('.cr'):
                    programFilepath = os.path.join(folder, filename)
                    program = ChatroomProgram(
                        programFilepath=programFilepath,
                        globalSkillset=self.globalSkillset,
                        globalContext=Context.globalContext
                    )
                    program.parse()
                    self.chatroomPrograms.append(program)
        
        if file:
            if file.endswith('.at'):
                program = AgentProgram(
                    programFilepath=file,
                    globalSkillset=self.globalSkillset,
                    globalContext=Context.globalContext
                )
                program.parse()
                self.agentPrograms.append(program)
            elif file.endswith('.cr'):
                program = ChatroomProgram(
                    programFilepath=file,
                    globalSkillset=self.globalSkillset,
                    globalContext=Context.globalContext
                )
                program.parse()
                self.chatroomPrograms.append(program)

        self.env = Env(
            globalContext=Context.globalContext,
            config=Context.globalContext.globalConfig,
            agentPrograms=self.agentPrograms,
            chatroomPrograms=self.chatroomPrograms,
            globalSkillset=self.globalSkillset
        )
        self.initContext = Context.create()

    def getAllAgents(self) -> dict[str, Agent]:
        return self.env.getAllAgents()

    def getAgent(self, name: str) -> Optional[Agent]:
        return self.env.getAgent(name)

    def getAllChatrooms(self) -> dict[str, Chatroom]:
        return self.env.getAllChatrooms()
    
    def getChatroom(self, name: str) -> Optional[Chatroom]:
        return self.env.getChatroom(name)

    def createContext(self, args: dict = {}) -> Context:
        context = self.initContext.copy()
        if "query" in args:
            context.setQuery(args["query"])
        return context

    def run(
            self, 
            context: Optional[Context] = None,
            chatroom: str = None, 
            agent: str = None, 
            args: dict = {}, 
            **kwargs) -> Context:
        if context is None:
            context = self.createContext(args)

        try:
            if chatroom:
                self.env.execute(
                    chatroomName=chatroom,
                    context=context,
                    query=context.getQueryStr(),
                    args=args,
                    **{**kwargs, "execNode" : context.getExecGraph().getRootNode()})
            elif agent:
                self.env.execute(
                    agentName=agent,
                    context=context,
                    query=context.getQueryStr(),
                    args=args,
                    **{**kwargs, "execNode" : context.getExecGraph().getRootNode()})
        except Exception as e:
            print(f"Engine run error: {str(e)}", flush=True)
            raise

        print(context.getExecGraph().dump())
        return context

    def executeAgent(
            self, 
            agentName: str, 
            code: str, 
            args: dict={}, 
            **kwargs):
        context = self.createContext(args)
        agent = self.env.agents[agentName]
        agent.setCodeAndCompile(code)
        return self.env.execute(
            agentName=agentName, 
            context=context, 
            args=args, 
            **{**kwargs, "execNode" : context.getExecGraph().getRootNode()})
