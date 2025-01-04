from milkie.context import Context


class DecisionModule():
    def __init__(self, engine=None):
        self.engine = engine

    def setEngine(self, engine):
        self.engine = engine

    def decide(
            self, 
            context :Context, 
            **kwargs):
        decisionResult = self.engine.synthesize(
            context.getCurQuery().query,
            context.retrievalResult)
        context.setDecisionResult(decisionResult)