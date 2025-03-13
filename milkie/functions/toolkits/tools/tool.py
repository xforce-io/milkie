from abc import abstractmethod


class Tool:

    @abstractmethod
    def execute(self, **args) -> str:
        pass

    def getToolName(self) -> str:
        return self.__class__.__name__
    
    