from abc import abstractmethod

from milkie.agent.base_agent import BaseAgent
from milkie.agent.team.deepqa import DeepQA
from milkie.agent.team.mrqa import MapReduceQA
from milkie.context import Context


class Strategy(object):
    
    @abstractmethod
    def getAgentName(self) -> str:
        pass

    def createAgent(self) -> BaseAgent:
        pass

class StrategyMRQA(Strategy):
    
    def getAgentName(self) -> str:
        self.agentName = "retrieval"
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return MapReduceQA(context, self.agentName)

class StrategyDeepQA(Strategy):
    
    def getAgentName(self) -> str:
        self.agentName = "qa"
        return self.agentName

    def createAgent(self, context :Context) -> BaseAgent:
        return DeepQA(context, self.agentName)