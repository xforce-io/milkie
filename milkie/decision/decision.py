from llama_index import QueryBundle


class DecisionModule():
    def __init__(self):
        pass

    def decide(self, context):
        decisionResult = context.engine.synthesize(
            context.getCurQuery(),
            context.retrievalResult)
        context.setDecisionResult(decisionResult)
        print(decisionResult)