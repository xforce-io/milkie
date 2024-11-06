from __future__ import annotations
from typing import List, Optional
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.runtime.global_toolkits import GlobalToolkits


class Toolbox(Toolkit):
    def __init__(self, globalToolkits :GlobalToolkits):
        self.toolkits = []
        self.globalToolkits = globalToolkits

    def merge(self, otherToolbox :Toolbox):
        self.toolkits.extend(otherToolbox.toolkits)

    def addToolkit(self, toolkit):
        self.toolkits.append(toolkit)

    def getTools(self):
        return [tool for toolkit in self.toolkits for tool in toolkit.getTools()]

    @staticmethod
    def createToolbox(globalToolkits :GlobalToolkits, toolkits :Optional[List[str]]=None):
        toolbox = Toolbox(globalToolkits)
        if not toolkits:
            toolkits = globalToolkits.getToolkitNames()
        
        for toolkit in toolkits:
            toolbox.addToolkit(globalToolkits.getToolkit(toolkit))
        return toolbox
