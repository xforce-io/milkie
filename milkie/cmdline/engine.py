from milkie.cmdline.cmdline import Cmdline
from milkie.cmdline.program_file import ProgramFile
from milkie.context import Context
from milkie.functions.toolkits.filesys_toolkits import FilesysToolKits
from milkie.global_context import GlobalContext
import logging

logger = logging.getLogger(__name__)

class Engine:
    def __init__(
            self,
            programFilePath: str,
            configPath: str = None) -> None:
        self.globalContext = GlobalContext.create(configPath)
        self.toolKits = {
            "FilesysToolKits": FilesysToolKits(),
        }
        self.program = ProgramFile(
            programFilepath=programFilePath,
            toolkitMaps=self.toolKits,
            globalContext=self.globalContext
        )
        self.program.parse()
        
        self.cmdline = Cmdline(
            context=Context(self.globalContext),
            config=self.globalContext.globalConfig,
            flowCode=self.program.getCode(),
            toolkit=self.program.toolkit)

    def run(self):
        return self.cmdline.execute(args={})