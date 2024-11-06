from typing import Dict
from milkie.functions.toolkits.toolkit import EmptyToolkit, Toolkit
from milkie.global_context import GlobalContext
import logging

from milkie.runtime.global_toolkits import GlobalToolkits
from milkie.runtime.program import Program, CollectorConfig, ContentCollector

logger = logging.getLogger(__name__)

class AgentProgram(Program):
    def __init__(
            self, 
            programFilepath: str,
            globalToolkits: GlobalToolkits = None,
            globalContext: GlobalContext = None
        ) -> None:
        super().__init__(programFilepath, globalToolkits, globalContext)

        self.systemPrompt = None
        self.code = None
        self.imports = []
        self.toolkit = None
        
        # 定义所有内容收集器的配置
        self.collectors = self.collectors | {
            'system': ContentCollector(CollectorConfig(
                prefix='@system',
                attributeName='systemPrompt',
                errorMsg="Agent system prompt is already set"
            )),
        }

    def getImports(self) -> list:
        return self.imports

    def getCode(self) -> str:
        if self.code is None:
            raise ValueError("Program has not been parsed yet")
        return self.code

    def getSystemPrompt(self) -> str:
        return self.systemPrompt

    def _handleSpecialLine(self, line: str) -> bool:
        if super()._handleSpecialLine(line):
            return True
        
        if line.startswith('@import'):
            self._handleImport(line)
            return True
        return False

    def _validateAndFinalize(self, parsedLines: list) -> None:
        super()._validateAndFinalize(parsedLines)

        # 处理工具包和代码
        self.toolkit = EmptyToolkit(self.globalContext) if len(self.imports) == 0 else Toolkit.getUnionToolkit(self.imports)
        self.code = '\n'.join(parsedLines)

    def _handleImport(self, line: str) -> None:
        toolkitName = line.split()[-1]
        if self.globalToolkits and self.globalToolkits.isValidToolkit(toolkitName):
            self.imports.append(self.globalToolkits.getToolkit(toolkitName))
            logger.debug(f"Imported toolkit: {toolkitName}")
        else:
            logger.warning(f"Toolkit not found: {toolkitName}")