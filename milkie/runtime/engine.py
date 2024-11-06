import os
from milkie.context import Context
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.env import Env
from milkie.runtime.global_toolkits import GlobalToolkits
from milkie.global_context import GlobalContext
import logging

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
            self,
            programFolder: str = None,
            programFilepath: str = None,
            configPath: str = None) -> None:
        self.globalContext = GlobalContext.create(configPath)
        self.globalToolkits = GlobalToolkits(self.globalContext)

        self.agentPrograms = []
        self.chatroomPrograms = []
        if programFolder:
            for filename in os.listdir(programFolder):
                if filename.endswith('.at'):
                    programFilepath = os.path.join(programFolder, filename)
                    program = AgentProgram(
                        programFilepath=programFilepath,
                        globalToolkits=self.globalToolkits,
                        globalContext=self.globalContext
                    )
                    program.parse()
                    self.agentPrograms.append(program)
                elif filename.endswith('.cr'):
                    programFilepath = os.path.join(programFolder, filename)
                    program = ChatroomProgram(
                        programFilepath=programFilepath,
                        globalToolkits=self.globalToolkits,
                        globalContext=self.globalContext
                    )
                    program.parse()
                    self.chatroomPrograms.append(program)
        
        if programFilepath:
            if programFilepath.endswith('.at'):
                program = AgentProgram(
                    programFilepath=programFilepath,
                    globalToolkits=self.globalToolkits,
                    globalContext=self.globalContext
                )
                program.parse()
                self.agentPrograms.append(program)
            elif programFilepath.endswith('.cr'):
                program = ChatroomProgram(
                    programFilepath=programFilepath,
                    globalToolkits=self.globalToolkits,
                    globalContext=self.globalContext
                )
                program.parse()
                self.chatroomPrograms.append(program)
        
        self.env = Env(
            context=Context(self.globalContext),
            config=self.globalContext.globalConfig,
            agentPrograms=self.agentPrograms,
            chatroomPrograms=self.chatroomPrograms,
            globalToolkits=self.globalToolkits
        )

    def run(self, chatroom: str = None, agent: str = None, args: dict = {}, **kwargs):
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