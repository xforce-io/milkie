from typing import List
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.runtime.global_toolkits import GlobalToolkits


class Toolbox(Toolkit):
    def __init__(self, globalToolkits :GlobalToolkits):
        self.toolkits = []
        self.globalToolkits = globalToolkits

    def addToolkit(self, toolkit):
        self.toolkits.append(toolkit)

    def getTools(self):
        return [tool for toolkit in self.toolkits for tool in toolkit.getTools()]

    @staticmethod
    def createToolbox(globalToolkits :GlobalToolkits, toolkits :List[str]):
        toolbox = Toolbox(globalToolkits)
        for toolkit in toolkits:
            toolbox.addToolkit(globalToolkits.getToolkit(toolkit))
        return toolbox
