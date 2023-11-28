from action import ActionModule
from decision import DecisionModule
from grounding import GroundingModule
from reasoning import ReasoningModule
from retrieval.retrieval import RetrievalModule
from context import Context

class Agent:
    def __init__(self):
        self.working_memory = {}
        self.long_term_memory = {}  # 根据需要设置
        self.grounding_module = GroundingModule()
        self.retrieval_module = RetrievalModule(self.long_term_memory)
        self.reasoning_module = ReasoningModule()
        self.decision_module = DecisionModule()
        self.action_module = ActionModule()

    def talk(self, query):
        context = Context(query)

    def processRound(self, context):
        pass

    def __grounding(self, json_data):
        self.grounding_module.update(json_data)
        self.working_memory.update(self.grounding_module.get_data())

    def __retrieval(self, context):
        return self.retrieval_module.retrieve(context)

    def __reasoning(self, context):
        retrieved_data = self.retrieval(context)
        return self.reasoning_module.reason(context, retrieved_data)

    def __decision(self, context):
        reasoning_info = self.reasoning(context)
        return self.decision_module.decide(context, reasoning_info)

    def __action(self, context, instructions):
        return self.action_module.act(instructions)

if __name__ == "__main__":
    # agent = Agent()
    # json_data = ...
    # agent.grounding(json_data)
    # instructions = agent.decision(context)
    # agent.action(instructions)