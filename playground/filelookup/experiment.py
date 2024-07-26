from milkie.strategy import Strategy
from playground.base_experiment import experiment, ex

@ex.automain
def mainFunc(
        strategy,
        benchmarks):
    kwargs = {
        "strategy":Strategy.getStrategy(strategy),
        "benchmarks":benchmarks
    }
    experiment(**kwargs)