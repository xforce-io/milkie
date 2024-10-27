from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
import logging

from milkie.runtime.global_toolkits import GlobalToolkits

logger = logging.getLogger(__name__)

class AgentProgram:
    def __init__(
            self, 
            programFilepath: str,
            globalToolkits: GlobalToolkits = None,
            globalContext: GlobalContext = None
        ) -> None:
        self.programFilepath = programFilepath
        self.globalToolkits = globalToolkits
        self.globalContext = globalContext
        self.name = None
        self.desc = None
        self.systemPrompt = None
        self.code = None
        self.imports = []
        self.toolkit = None
        self.program = self._readProgramFile()

    def _readProgramFile(self) -> str:
        try:
            with open(self.programFilepath, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Program file read successfully: {self.programFilepath}")
            return content
        except IOError as e:
            logger.error(f"Failed to read program file: {self.programFilepath}")
            raise IOError(f"Unable to read program file: {e}")

    def parse(self) -> None:
        lines = self.program.split('\n')
        parsedLines = []

        self.desc = None
        inDesc = False
        descLines = []
        
        self.systemPrompt = None
        inSystemPrompt = False
        systemPromptLines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('@import'):
                self._handleImport(line)
            elif line.startswith('@name'):
                self.name = line.split()[-1].strip()
                if self.name == "":
                    raise SyntaxError(f"Agent name is empty[{self.programFilepath}]")
            elif line.startswith('@desc'):
                if self.desc is not None:
                    raise SyntaxError(f"Agent description is already set[{self.programFilepath}]")
                
                inDesc = not inDesc
                if inDesc:
                    line = line[len('@desc'):].strip()
                    if line != "":
                        self.desc = line
                continue
            elif inDesc and self.desc is None:
                descLines.append(line)
            elif line.startswith('@system'):
                if self.systemPrompt is not None:
                    raise SyntaxError(f"Agent system prompt is already set[{self.programFilepath}]")
                
                inSystemPrompt = not inSystemPrompt
                if inSystemPrompt:
                    line = line.split('@system')[1].strip()
                    if line != "":
                        self.systemPrompt = line
                continue
            elif inSystemPrompt and self.systemPrompt is None:
                systemPromptLines.append(line)
            elif not line.startswith('//'):
                parsedLines.append(line)
        
        if self.name is None:
            raise SyntaxError(f"Agent name is not set[{self.programFilepath}]")
        
        self.toolkit = Toolkit.getUnionToolkit(self.imports)
        self.code = '\n'.join(parsedLines)

        if self.systemPrompt is None:
            self.systemPrompt = '\n'.join(systemPromptLines).strip()

        if self.desc is None:
            self.desc = '\n'.join(descLines).strip()

        if self.desc == "":
            raise SyntaxError(f"Agent description is empty[{self.programFilepath}]")
        
        logger.info("Program file parsed successfully")

    def _handleImport(self, line: str) -> None:
        toolkitName = line.split()[-1]
        if self.globalToolkits and self.globalToolkits.isValidToolkit(toolkitName):
            self.imports.append(self.globalToolkits.getToolkit(toolkitName))
            logger.debug(f"Imported toolkit: {toolkitName}")
        else:
            logger.warning(f"Toolkit not found: {toolkitName}")

    def getImports(self) -> list:
        return self.imports

    def getCode(self) -> str:
        if self.code is None:
            raise ValueError("Program has not been parsed yet")
        return self.code

    def getSystemPrompt(self) -> str:
        return self.systemPrompt
    
    def getDesc(self) -> str:
        return self.desc
