from milkie.runtime.program import Program
from milkie.global_context import GlobalContext
from milkie.functions.toolkits.skillset import Skillset

class ChatroomProgram(Program):
    def __init__(
            self, 
            programFilepath: str, 
            globalSkillset: Skillset = None, 
            globalContext: GlobalContext = None
        ) -> None:
        super().__init__(programFilepath, globalSkillset, globalContext)

        self.host = None
        self.prologue = None

    def _handleSpecialLine(self, line: str) -> bool:
        if super()._handleSpecialLine(line):
            return True
        
        if line.startswith('@host'):
            self.host = line.split()[-1].strip()
            if not self.host:
                raise SyntaxError(f"Host is not set[{self.programFilepath}]")
            return True

        if line.startswith('@prologue'):
            self.prologue = line.split()[-1].strip()
            if not self.prologue:
                raise SyntaxError(f"Prologue is not set[{self.programFilepath}]")
            return True
        return False
