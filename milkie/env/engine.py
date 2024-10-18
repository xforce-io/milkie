import os
from milkie.context import Context
from milkie.env.agent_program import AgentProgram
from milkie.env.env import Env
from milkie.functions.toolkits.filesys_toolkit import FilesysToolKit
from milkie.functions.toolkits.basic_toolkit import BasicToolKit
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
        self.toolKits = {
            "FilesysToolKit": FilesysToolKit(self.globalContext),
            "BasicToolKit": BasicToolKit(self.globalContext),
        }

        self.agentPrograms = []
        if programFolder:
            for filename in os.listdir(programFolder):
                if filename.endswith('.mk'):
                    programFilePath = os.path.join(programFolder, filename)
                    program = AgentProgram(
                        programFilepath=programFilePath,
                        toolkitMaps=self.toolKits,
                        globalContext=self.globalContext
                    )
                    program.parse()
                    self.agentPrograms.append(program)
        
        if programFilepath:
            program = AgentProgram(
                programFilepath=programFilepath,
                toolkitMaps=self.toolKits,
                globalContext=self.globalContext
            )
            program.parse()
            self.agentPrograms.append(program)
        
        self.env = Env(
            context=Context(self.globalContext),
            config=self.globalContext.globalConfig,
            agentPrograms=self.agentPrograms)

    def run(self, agent: str = None, args: dict = {}):
        return self.env.execute(
            query=args["query_str"] if "query_str" in args else None, 
            args=args, 
            agentName=agent)
