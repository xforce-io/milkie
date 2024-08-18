from abc import abstractmethod
from typing import Any

from milkie.agent.base_agent import BaseAgent
from milkie.agent.prompt_agent import PromptAgent
from milkie.agent.team.deepqa import DeepQA
from milkie.agent.team.file_lookup import FileLookupAgent
from milkie.agent.team.mrqa import MapReduceQA
from milkie.context import Context

class Strategy(object):
    
    @abstractmethod
    def getAgentName(self) -> str:
        pass

    @abstractmethod
    def createAgent(self) -> BaseAgent:
        pass

    def __str__(self) -> str:
        pass

    def getStrategy(name :str) -> Any:
        if name == "raw":
            return StrategyRaw()
        elif name == "mrqa":
            return StrategyMRQA()
        elif name == "deepqa":
            return StrategyDeepQA()
        elif name == "file_lookup":
            return StrategyFileLookup()
        else:
            raise ValueError(f"Unknown strategy name: {name}")

class StrategyMRQA(Strategy):

    def __init__(self) -> None:
        self.agentName = "retrieval"
    
    def getAgentName(self) -> str:
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return MapReduceQA(context, self.agentName)

    def __str__(self) -> str:
        return "MRQA"

class StrategyDeepQA(Strategy):

    def __init__(self) -> None:
        self.agentName = "qa"
    
    def getAgentName(self) -> str:
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return DeepQA(context, self.agentName)

    def __str__(self) -> str:
        return "DeepQA"

class StrategyRaw(Strategy):

    def __init__(self) -> None:
        self.agentName = "raw"
    
    def getAgentName(self) -> str:
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return PromptAgent(context, None)

    def __str__(self) -> str:
        return "Raw"

class StrategyFileLookup(Strategy):

    def __init__(self) -> None:
        self.agentName = "file_lookup"
    
    def getAgentName(self) -> str:
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return FileLookupAgent(context)

    def __str__(self) -> str:
        return "FileLookup"