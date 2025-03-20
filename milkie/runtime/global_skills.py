from typing import List
from milkie.functions.toolkits.agent_toolkit import AgentToolkit
from milkie.functions.toolkits.basic_toolkit import BasicToolkit
from milkie.functions.toolkits.filesys_toolkit import FilesysToolkit
from milkie.functions.toolkits.search_toolkit import SearchToolkit
from milkie.functions.toolkits.skillset import Skillset
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.test_toolkit import TestToolkit

class GlobalSkills(object):
    def __init__(self, globalContext):
        self.globalContext = globalContext

        toolkits = [
            FilesysToolkit(self.globalContext),
            BasicToolkit(self.globalContext),
            SearchToolkit(self.globalContext),
            TestToolkit(self.globalContext),
        ]

        self.toolkits = {
            toolkit.getName(): toolkit
            for toolkit in toolkits
        }
        self.agents = dict()

    def getSkill(self, name: str) -> Toolkit:
        toolkit = self.toolkits.get(name)
        if toolkit:
            return toolkit
        
        agent = self.agents.get(name)
        if agent:
            return AgentToolkit(agent)
        
        raise RuntimeError(f"Skill not found: {name}")

    def getSkillNames(self):
        return list(self.toolkits.keys()) + list(self.agents.keys())

    def isValidSkillName(self, name: str):
        return name in self.toolkits or name in self.agents

    def createSkillset(self):
        skillset = Skillset()
        for toolkit in self.toolkits.values():
            skillset.addSkill(toolkit)
        for agent in self.agents.values():
            skillset.addSkill(AgentToolkit(agent))
        return skillset
