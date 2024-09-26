from milkie.functions.toolkits.base_toolkits import BaseToolkit
from milkie.global_context import GlobalContext
import logging

logger = logging.getLogger(__name__)

class ProgramFile:
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
        
        for line in lines:
            line = line.strip()
            if line.startswith('@import'):
                self._handleImport(line)
            elif not line.startswith('//'):
                parsedLines.append(line)
        
        if len(self.imports) == 0:
            raise SyntaxError("No toolkit imported")
        
        self.toolkit = BaseToolkit.getUnionToolkit(self.imports)
        self.code = '\n'.join(parsedLines)
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

