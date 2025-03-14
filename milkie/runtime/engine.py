import os
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
        self.globalContext = GlobalContext.create(config)
        self.globalSkills = GlobalSkills(self.globalContext)
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
                        globalContext=self.globalContext
                    )
                    program.parse()
                    self.agentPrograms.append(program)
                elif filename.endswith('.cr'):
                    programFilepath = os.path.join(folder, filename)
                    program = ChatroomProgram(
                        programFilepath=programFilepath,
                        globalSkillset=self.globalSkillset,
                        globalContext=self.globalContext
                    )
                    program.parse()
                    self.chatroomPrograms.append(program)
        
        if file:
            if file.endswith('.at'):
                program = AgentProgram(
                    programFilepath=file,
                    globalSkillset=self.globalSkillset,
                    globalContext=self.globalContext
                )
                program.parse()
                self.agentPrograms.append(program)
            elif file.endswith('.cr'):
                program = ChatroomProgram(
                    programFilepath=file,
                    globalSkillset=self.globalSkillset,
                    globalContext=self.globalContext
                )
                program.parse()
                self.chatroomPrograms.append(program)

        self.env = None
        self.initContext = Context(self.globalContext)

    def run(self, chatroom: str = None, agent: str = None, args: dict = {}, **kwargs):
        if self.env is None:
            self.env = Env(
                context=self.initContext,
                config=self.globalContext.globalConfig,
                agentPrograms=self.agentPrograms,
                chatroomPrograms=self.chatroomPrograms,
                globalSkillset=self.globalSkillset
            )

        try:
            if chatroom:
                return self.env.execute(
                    chatroomName=chatroom,
                    query=args["query"] if "query" in args else None, 
                    args=args, 
                    **kwargs)
            elif agent:
                return self.env.execute(
                    agentName=agent,
                    query=args["query"] if "query" in args else None, 
                    args=args, 
                    **kwargs)
        except Exception as e:
            print(f"Engine run error: {str(e)}", flush=True)
            raise

    def executeAgent(
            self, 
            agentName: str, 
            code: str, 
            args: dict={}, 
            **kwargs):
        agent = self.env.agents[agentName]
        agent.setCodeAndCompile(code)
        return self.env.execute(agentName=agentName, args=args, **kwargs)
