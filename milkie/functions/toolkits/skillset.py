from __future__ import annotations
from typing import List, Optional
from milkie.functions.toolkits.toolkit import Toolkit

class Skillset(Toolkit):
    def __init__(self):
        self.skillset = []

    def merge(self, otherSkillset :Skillset):
        self.skillset.extend(otherSkillset.skillset)

    def addSkill(self, skill :Toolkit):
        if skill.getName() not in self.getSkillNames():
            self.skillset.append(skill)

    def getSkill(self, skillName :str) -> Toolkit:
        for skill in self.skillset:
            if skill.getName() == skillName:
                return skill
        return None 

    def getSkillNames(self):
        return [skill.getName() for skill in self.skillset]

    def getTools(self):
        return [tool for skill in self.skillset for tool in skill.getTools()]

    @staticmethod
    def createSkillset(globalSkillset :Skillset, skillNames :Optional[List[str]]=None):
        newSkillset = Skillset()
        if not skillNames:
            return globalSkillset
        
        for skillName in skillNames:
            skill = globalSkillset.getSkill(skillName)
            if skill:
                newSkillset.addSkill(skill)
        return newSkillset


    def isEmpty(self) -> bool:
        return super().isEmpty()    