from dataclasses import dataclass
from typing import Dict, Optional
import logging
from milkie.global_context import GlobalContext
from milkie.runtime.global_toolkits import GlobalToolkits

logger = logging.getLogger(__name__)

@dataclass
class CollectorConfig:
    """内容收集器的配置"""
    prefix: str
    attributeName: str
    errorMsg: str

class ContentCollector:
    """内容收集器，用于处理多行内容的收集"""
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.isCollecting = False
        self.lines = []
        self.content = None

    def handleStart(self, line: str, currentValue: Optional[str], filePath: str) -> None:
        """处理收集开始的标记行"""
        if currentValue is not None:
            raise SyntaxError(f"{self.config.errorMsg}[{filePath}]")
        
        self.isCollecting = not self.isCollecting
        content = line[len(self.config.prefix):].strip()
        if content:
            self.content = content

    def addLine(self, line: str) -> None:
        """添加一行内容"""
        if self.isCollecting and self.content is None:
            self.lines.append(line)

    def getFinalContent(self) -> str:
        """获取最终内容"""
        if self.content is None:
            self.content = '\n'.join(self.lines).strip()
        return self.content

class Program:
    def __init__(
            self, 
            programFilepath: str,
            globalToolkits: GlobalToolkits = None,
            globalContext: GlobalContext = None
        ):
        self.programFilepath = programFilepath
        self.globalToolkits = globalToolkits
        self.globalContext = globalContext

        self.name = None
        self.desc = None
        self.experts = None
        self.program = self.readProgramFile()

        # 定义所有内容收集器的配置
        self.collectors: Dict[str, ContentCollector] = {
            'desc': ContentCollector(CollectorConfig(
                prefix='@desc',
                attributeName='desc',
                errorMsg="Program description is already set"
            )),
            'experts': ContentCollector(CollectorConfig(
                prefix='@experts',
                attributeName='experts',
                errorMsg="Program experts is already set"
            ))
        }

    def readProgramFile(self) -> str:
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
        
        for line in lines:
            if not line or line.startswith('//'):
                continue

            if self._handleSpecialLine(line):
                continue

            parsedLines.append(line)

        self._validateAndFinalize(parsedLines)
        logger.info("Program file parsed successfully")
 
    def getDesc(self) -> str:
        return self.desc

    def getExpertAssignments(self) -> list[tuple[str, str]]:
        if len(self.experts.strip()) == 0:
            return None

        experts = []
        pairs = [expert.strip() for expert in self.experts.split("\n") if len(expert.strip()) > 0]
        for pair in pairs:
            roleAndName = pair.split("->")
            if len(roleAndName) == 2:
                role, name = roleAndName
                experts.append((role.strip(), name.strip()))
            elif len(roleAndName) == 1:
                name = roleAndName[0].strip()
                experts.append((name, name))
            else:
                raise SyntaxError(f"Invalid expert format[{self.programFilepath}]")
        return experts

    def _handleName(self, line: str) -> None:
        self.name = line.split()[-1].strip()
        if not self.name:
            raise SyntaxError(f"Program name is empty[{self.programFilepath}]")

    def _handleSpecialLine(self, line: str) -> bool:
        if line.startswith('@name'):
            self._handleName(line)
            return True

        # 处理所有收集器
        for collector in self.collectors.values():
            if line.startswith(collector.config.prefix):
                collector.handleStart(line, getattr(self, collector.config.attributeName), self.programFilepath)
                return True
            if collector.isCollecting:
                collector.addLine(line)
                return True

        return False

    def _validateAndFinalize(self, parsedLines: list) -> None:
        if self.name is None:
            raise SyntaxError(f"Program name is not set[{self.programFilepath}]")

        # 处理所有收集器的内容
        for collector in self.collectors.values():
            content = collector.getFinalContent()
            if collector.config.attributeName == 'desc' and not content:
                raise SyntaxError(f"Program description is empty[{self.programFilepath}]")
            setattr(self, collector.config.attributeName, content)
