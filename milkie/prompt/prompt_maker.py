from milkie.functions.toolkits.base import BaseToolkit


class PromptMaker:

    def __init__(self, toolkit :BaseToolkit) -> None:
        self.task :str = None
        self.toolkit = toolkit

    def setTask(self, task :str):
        self.task = task

    def getToolsDesc(self) -> str:
        return self.toolkit.getToolsDesc()
