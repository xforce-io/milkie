from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
import logging

logger = logging.getLogger(__name__)

class AgentProgram:
    def __init__(
            self, 
            programFilepath: str,
            toolkitMaps: dict = None,
            globalContext: GlobalContext = None
        ) -> None:
        """
        初始化 ProgramFile 对象

        :param programFilepath: 程序文件的路径
        :param toolkits: 工具包字典
        :param globalContext: 全局上下文对象
        """
        self.programFilepath = programFilepath
        self.toolkitMaps = toolkitMaps
        self.globalContext = globalContext
        self.name = None
        self.systemPrompt = None
        self.code = None
        self.imports = []
        self.toolkit = None
        self.program = self._readProgramFile()

    def _readProgramFile(self) -> str:
        """
        读取程序文件内容

        :return: 文件内容字符串
        """
        try:
            with open(self.programFilepath, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Program file read successfully: {self.programFilepath}")
            return content
        except IOError as e:
            logger.error(f"Failed to read program file: {self.programFilepath}")
            raise IOError(f"Unable to read program file: {e}")

    def parse(self) -> None:
        """
        解析程序文件内容，提取导入的工具包并处理代码
        """
        lines = self.program.split('\n')
        parsedLines = []
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
            elif line.startswith('@system'):
                inSystemPrompt = not inSystemPrompt
                continue
            elif inSystemPrompt:
                systemPromptLines.append(line)
            elif not line.startswith('//'):
                parsedLines.append(line)
        
        if self.name is None:
            raise SyntaxError(f"Agent name is not set[{self.programFilepath}]")
        
        self.toolkit = Toolkit.getUnionToolkit(self.imports)
        self.code = '\n'.join(parsedLines)
        self.systemPrompt = '\n'.join(systemPromptLines).strip()
        logger.info("Program file parsed successfully")

    def _handleImport(self, line: str) -> None:
        """
        处理导入语句

        :param line: 导入语句行
        """
        toolkitName = line.split()[-1]
        if self.toolkitMaps and toolkitName in self.toolkitMaps:
            self.imports.append(self.toolkitMaps[toolkitName])
            logger.debug(f"Imported toolkit: {toolkitName}")
        else:
            logger.warning(f"Toolkit not found: {toolkitName}")

    def getImports(self) -> list:
        """
        获取导入的工具包列表

        :return: 导入的工具包列表
        """
        return self.imports

    def getCode(self) -> str:
        """
        获取处理后的代码

        :return: 处理后的代码字符串
        """
        if self.code is None:
            raise ValueError("Program has not been parsed yet")
        return self.code

    def getSystemPrompt(self) -> str:
        """
        获取系统提示

        :return: 系统提示字符串
        """
        return self.systemPrompt
