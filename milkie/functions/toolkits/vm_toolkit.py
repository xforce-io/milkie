from typing import List

from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.openai_function import OpenAIFunction
from milkie.global_context import GlobalContext

class VMToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

        self.queryAsArg = True

    def getName(self) -> str:
        return "VMToolkit"
    
    def execBash(self, cmd: str) -> str:
        """
        在虚拟机中执行bash命令, 并返回执行结果
        
        Args:
            cmd (str): 要执行的bash命令

        Returns:
            str: 执行结果
        """
        return self.globalContext.vm.execBash(cmd)
    
    def execPython(self, cmd: str) -> str:
        """
        在虚拟机中执行python命令, 并返回执行结果
        
        Args:
            cmd (str): 要执行的python命令

        Returns:
            str: 执行结果
        """
        return self.globalContext.vm.execPython(cmd)
    

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.execBash),
            OpenAIFunction(self.execPython),
        ]
    