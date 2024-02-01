from milkie.action import ActionModule
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.decision.decision import DecisionModule
from milkie.grounding import GroundingModule
from milkie.reasoning import ReasoningModule
from milkie.retrieval.retrieval import RetrievalModule

class Agent:
    def __init__(self, globalConfig :GlobalConfig):
        self.globalConfig = globalConfig 
        self.context = Context(self.globalConfig)
        self.working_memory = {}
        self.long_term_memory = {}  # 根据需要设置
        self.grounding_module = GroundingModule()
        self.retrieval_module = RetrievalModule(context=self.context)
        self.reasoning_module = ReasoningModule()
        self.decision_module = DecisionModule()
        self.action_module = ActionModule()

    def task(self, query):
        self.context.setCurQuery(query)
        self.processRound(self.context)
        while not self._end():
            self.processRound(self.context)
        return self.context.decisionResult

    def processRound(self, context):
        self.__retrieval(context)
        self.__reasoning(context)
        self.__decision(context)
    
    def _end(self) -> bool:
        return True

    def __retrieval(self, context):
        return self.retrieval_module.retrieve(context)

    def __reasoning(self, context):
        return self.reasoning_module.reason(context)

    def __decision(self, context):
        return self.decision_module.decide(context)

    def __action(self, context, instructions):
        return self.action_module.act(instructions)

    def __grounding(self, json_data):
        self.grounding_module.update(json_data)
        self.working_memory.update(self.grounding_module.get_data())

if __name__ == "__main__":
    agent = Agent(GlobalConfig("config/global.yaml"))
    agent.task("你好")