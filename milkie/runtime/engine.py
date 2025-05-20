import os
from typing import Optional
from milkie.agent.agent import Agent
from milkie.chatroom.chatroom import Chatroom
from milkie.context import Context
from milkie.runtime.agent_program import AgentProgram
from milkie.runtime.chatroom_program import ChatroomProgram
from milkie.runtime.env import Env
from milkie.runtime.global_skills import GlobalSkills
from milkie.global_context import GlobalContext
import logging

from milkie.types.object_type import ObjectTypeFactory

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
        self.globalObjectTypes = ObjectTypeFactory()

        self.agentPrograms = []
        self.chatroomPrograms = []

        if folder:
            self._scanAndLoadPrograms(folder)
        if file:
            self._scanAndLoadPrograms(file)

        self.env = Env(
            globalContext=Context.globalContext,
            config=Context.globalContext.globalConfig,
            agentPrograms=self.agentPrograms,
            chatroomPrograms=self.chatroomPrograms,
            globalSkillset=self.globalSkillset,
            globalObjectTypes=self.globalObjectTypes
        )
        self.initContext = Context.create()

    def _loadProgram(self, programFilepath: str):
        """Loads and parses a single agent or chatroom program file."""
        try:
            if programFilepath.endswith('.at'):
                program = AgentProgram(
                    programFilepath=programFilepath,
                    globalSkillset=self.globalSkillset,
                    globalContext=Context.globalContext
                )
                program.parse()
                self.agentPrograms.append(program)
                logger.info(f"Loaded agent program: {programFilepath}")
            elif programFilepath.endswith('.cr'):
                program = ChatroomProgram(
                    programFilepath=programFilepath,
                    globalSkillset=self.globalSkillset,
                    globalContext=Context.globalContext
                )
                program.parse()
                self.chatroomPrograms.append(program)
                logger.info(f"Loaded chatroom program: {programFilepath}")
            elif programFilepath.endswith('.type'):
                self.globalObjectTypes.load(programFilepath)
                logger.info(f"Loaded object type: {programFilepath}")
            # Files with other extensions are ignored silently
        except Exception as e:
            logger.error(f"Failed to load or parse program '{programFilepath}': {e}", exc_info=True)
            # Decide if loading should stop on error or just skip the file
            # For now, we log and continue.

    def _scanAndLoadPrograms(self, path: str):
        """Recursively scans a path (file or directory) and loads programs."""
        if not os.path.exists(path):
            logger.warning(f"Path specified for loading programs does not exist: {path}")
            return

        if os.path.isfile(path):
            self._loadProgram(path)
        elif os.path.isdir(path):
            try:
                for filename in os.listdir(path):
                    fullItemPath = os.path.join(path, filename)
                    # Recursive call for subdirectories or files
                    self._scanAndLoadPrograms(fullItemPath)
            except PermissionError:
                logger.warning(f"Permission denied to access directory: {path}")
            except Exception as e:
                logger.error(f"Error listing directory '{path}': {e}", exc_info=True)

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
        else:
            context.setQuery(None)
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
        
        with open("log/exec_graph.txt", "w") as f:
            execGraph = context.getExecGraph().dump()
            f.write(execGraph)
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
