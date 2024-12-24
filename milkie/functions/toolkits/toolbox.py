from __future__ import annotations
from typing import List, Optional
from milkie.context import Context
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
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

    def queryExpert(self, query: str, context: Optional[Context] = None) -> str:
        for toolkit in self.toolkits:
            if isinstance(toolkit, AgentToolkit) and query[1:].startswith(toolkit.agent.name):
                responce = toolkit.agent.execute(
                    query=query, 
                    args=context.getVarDict().getGlobalDict())
                return responce.respStr
        return ""

    def isEmpty(self) -> bool:
        return super().isEmpty()    