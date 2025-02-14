from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.basic_toolkit import BasicToolkit
from milkie.functions.toolkits.filesys_toolkit import FilesysToolkit
from milkie.functions.toolkits.search_toolkit import SearchToolkit
from milkie.functions.toolkits.toolkit import Toolkit


class GlobalToolkits(object):
    def __init__(self, globalContext):
        self.globalContext = globalContext
        self.toolkits = {
            "FilesysToolkit": FilesysToolkit(self.globalContext),
            "BasicToolkit": BasicToolkit(self.globalContext),
            "SearchToolkit": SearchToolkit(self.globalContext),
        }
        self.agents = dict()

    def addAgent(self, agent):
        self.agents[agent.name] = agent

    def getToolkit(self, name: str) -> Toolkit:
        toolkit = self.toolkits.get(name)
        if toolkit:
            return toolkit
        
        agent = self.agents.get(name)
        if agent:
            return AgentToolkit(agent)
        
        raise RuntimeError(f"Toolkit not found: {name}")

    def getToolkitNames(self):
        return list(self.toolkits.keys()) + list(self.agents.keys())

    def isValidToolkit(self, name: str):
        return name in self.toolkits or name in self.agents
