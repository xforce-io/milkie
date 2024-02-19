from milkie.context import Context


class DecisionModule():
    def __init__(self, engine):
        self.engine = engine

    def decide(self, context :Context):
        decisionResult = self.engine.synthesize(
            context.getCurQuery(),
            context.retrievalResult)
        context.setDecisionResult(decisionResult)
        print(decisionResult)