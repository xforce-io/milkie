from milkie.functions.toolkits.toolkit import Toolkit


class PromptMaker:

    def __init__(self, toolkit :Toolkit) -> None:
        self.task :str = None
        self.toolkit = toolkit

    def setTask(self, task :str):
        self.task = task

    def getToolsDesc(self) -> str:
        return self.toolkit.getToolsDesc()
