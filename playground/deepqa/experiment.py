import time, logging

from llama_index.core import Response

from milkie.benchmark.benchtype import BenchTypeKeyword, Benchmarks
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.strategy import Strategy

from milkie.utils.commons import getMemStat
from playground.global_config import makeGlobalConfig
from playground.init import createExperiment

logger = logging.getLogger(__name__)

ex = createExperiment()

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
        resps = []
        try:
            resps = agent.taskBatch(
                prompt, 
                argsList, 
                **globalConfig.getLLMConfig().generationArgs.toJson())
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error[{e}] in agentTaskBatch")

        t1 = time.time()
        lenOutputs += sum(len(resp.response) for resp in resps)
        if len(resps) >= 0 and "numTokens" in resps[0].metadata:
            totalTokens += sum(resp.metadata["numTokens"] for resp in resps)
        totalTime += t1-t0
        numQueries += len(resps)
        numBatches += 1
        return resps 

    benchmarks.evalAndReport(agent=agentTaskBatch, prompt="{query_str}")
    tokensPerSec = float(totalTokens)/totalTime

    #TODO: 1.412 is observered from the A800 GPU, need to remove this hard code
    ex.log_scalar("total", numQueries)
    ex.log_scalar("costSec", totalTime)
    ex.log_scalar("avgOutputLen", lenOutputs/numQueries)
    ex.log_scalar("avgQueryLatSec", totalTime/numQueries)
    ex.log_scalar("avgBatchLatSec", totalTime/numBatches)
    ex.log_scalar("tokensPerSec", tokensPerSec)
    ex.log_scalar("memory", globalContext.settings.llm.getMem())
    ex.log_scalar("mbu", globalContext.settings.llm.getMBU(tokensPerSec, 1.412 * 1024**4))

    getMemStat()

@ex.automain
def mainFunc(
        type,
        strategy,
        llm_model,
        framework,
        ctx_len,
        rewrite_strategy,
        chunk_augment,
        benchmarks):
    kwargs = {
        "type":type,
        "strategy":Strategy.getStrategy(strategy),
        "llm_model":llm_model,
        "framework":framework,
        "ctx_len":ctx_len,
        "rewrite_strategy":rewrite_strategy,
        "chunk_augment":chunk_augment,
        "reranker":"FLAGEMBED",
        "rerank_position":"NONE",
        "channel_recall":30,
        "similarity_top_k":20,
        "benchmarks":benchmarks
    }
    experiment(**kwargs)