import time, logging
from sacred import Experiment

from llama_index.core import Response

from milkie.benchmark.benchtype import BenchTypeKeyword, Benchmarks
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.strategy import Strategy

from playground.global_config import makeGlobalConfig

logger = logging.getLogger(__name__)

from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))

@ex.capture()
def experiment(
        strategy :Strategy,
        **kwargs):
    globalConfig = makeGlobalConfig(strategy, **kwargs)

    modelFactory = ModelFactory()
    globalContext = GlobalContext(globalConfig, modelFactory)
    context = Context(globalContext=globalContext)
    agent = strategy.createAgent(context)

    benchmarks = []
    for benchmark in kwargs["benchmarks"].split(";"):
        if len(benchmark.strip()) == 0:
            continue
        benchmarks.append(BenchTypeKeyword(benchmark.strip()))

    benchmarks = Benchmarks(
        ex,
        benchmarks,
        globalConfig.getLLMConfig().batchSize)

    numQueries = 0
    lenOutputs = 0
    numBatches = 0
    totalTime = 0
    totalTokens = 0

    def agentTaskBatch(prompt :str, argsList :list) -> list[Response]:
        nonlocal agent, numQueries, lenOutputs, numBatches, totalTime, totalTokens
        t0 = time.time()
        resps = agent.taskBatch(
            prompt, 
            argsList, 
            **globalConfig.getLLMConfig().generationArgs.toJson())
        t1 = time.time()
        lenOutputs += sum(len(resp.response) for resp in resps)
        if len(resps) >= 0 and "numTokens" in resps[0].metadata:
            totalTokens += sum(resp.metadata["numTokens"] for resp in resps)
        totalTime += t1-t0
        numQueries += len(resps)
        numBatches += 1
        return resps 

    benchmarks.evalAndReport(agent=agentTaskBatch, prompt="{query_str}")

@ex.automain
def mainFunc(
        strategy,
        llm_model,
        framework,
        rewrite_strategy,
        benchmarks):
    kwargs = {
        "strategy":Strategy.getStrategy(strategy),
        "llm_model":llm_model,
        "framework":framework,
        "rewrite_strategy":rewrite_strategy,
        "reranker":"FLAGEMBED",
        "rerank_position":"NONE",
        "channel_recall":30,
        "similarity_top_k":20,
        "benchmarks":benchmarks
    }
    experiment(**kwargs)