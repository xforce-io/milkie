from abc import abstractmethod

from milkie.agent.base_agent import BaseAgent
from milkie.agent.team.deepqa import DeepQA
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

class StrategyMRQA(Strategy):
    
    def getAgentName(self) -> str:
        self.agentName = "retrieval"
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return MapReduceQA(context, self.agentName)

    def __str__(self) -> str:
        return "MRQA"

class StrategyDeepQA(Strategy):
    
    def getAgentName(self) -> str:
        self.agentName = "qa"
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return DeepQA(context, self.agentName)

    def __str__(self) -> str:
        return "DeepQA"