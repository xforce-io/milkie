import time, logging
from sacred import Experiment

from llama_index.legacy.response.schema import Response

from milkie.benchmark.benchtype import BenchTypeKeyword, Benchmarks
from milkie.config.config import FRAMEWORK
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.prompt.prompt import Loader
from milkie.strategy import Strategy, StrategyRaw
from milkie.utils.commons import getMemStat
from playground.global_config import makeGlobalConfig

logger = logging.getLogger(__name__)

ModelYi34 = "/mnt/data1/.cache/huggingface/hub/01ai/Yi-34B-Chat/"

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelBaichuan13bChat = Prefix+"baichuan-inc/Baichuan2-13B-Chat"
ModelQwen14bChat = Prefix+"qwen/Qwen-14B-Chat"
ModelQwenV15S14bChat = Prefix+"qwen/Qwen1.5-14B-Chat/"
ModelQwenV15S14bGPTQINT4Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int4/"
ModelQwenV15S14bGPTQINT8Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int8/"
ModelQwenV15S14bAWQChat = Prefix+"qwen/Qwen1___5-14B-Chat-AWQ/"

from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))
modelFactory = ModelFactory()

promptQA = Loader.load("qa_init")

@ex.capture()
def experiment(
        strategy :Strategy,
        **kwargs):
    globalConfig = makeGlobalConfig(**kwargs)
    globalConfig.memoryConfig = None

    globalContext = GlobalContext(globalConfig, modelFactory)
    context = Context(globalContext=globalContext)
    agent = strategy.createAgent(context)

    benchmarks = Benchmarks([
            #BenchTypeKeyword("benchmark/410_key.jsonl"),
            BenchTypeKeyword("benchmark/fd100_key.jsonl"),
        ],
        globalConfig.getLLMConfig().batchSize)

    numQueries = 0
    numBatches = 0
    totalTime = 0
    totalTokens = 0

    def agentTaskBatch(prompt :str, argsList :list) -> list[Response]:
        nonlocal agent, numQueries, numBatches, totalTime, totalTokens
        t0 = time.time()
        resps = agent.taskBatch(
            prompt, 
            argsList, 
            **globalConfig.getLLMConfig().generationArgs.toJson())
        t1 = time.time()
        totalTokens += sum(resp.metadata["numTokens"] for resp in resps)
        totalTime += t1-t0
        numQueries += len(resps)
        numBatches += 1
        return resps 

    def agentTaskSingle(prompt :str, argsList :list) -> list[Response]:
        nonlocal agent, numQueries, numBatches, totalTime, totalTokens
        t0 = time.time()
        resp = agent.task(
            prompt, 
            argsList, 
            **globalConfig.getLLMConfig().generationArgs.toJson())
        t1 = time.time()
        totalTokens += resp.metadata["numTokens"]
        totalTime += t1-t0
        numQueries += 1
        numBatches += 1
        return [resp]

    benchmarks.evalAndReport(agent=agentTaskBatch, prompt=promptQA)
    tokensPerSec = float(totalTokens)/totalTime

    #TODO: 1.412 is observered from the A800 GPU, need to remove this hard code
    logger.info(f"Running "
                f"kwargs[{kwargs}] "
                f"costSec[{totalTime}] "
                f"avgQueryLatSec[{totalTime/numQueries}] "
                f"avgBatchLatSec[{totalTime/numBatches}] "
                f"tokensPerSec[{tokensPerSec}] "
                f"memory[{globalContext.settings.llm.getMem()}] "
                f"mbu[{globalContext.settings.llm.getMBU(tokensPerSec, 1.412 * 1024**4)}] ")
    ex.log_scalar("total", numQueries)
    ex.log_scalar("avgQueryLatSec", totalTime/numQueries)
    ex.log_scalar("avgBatchLatSec", totalTime/numBatches)
    ex.log_scalar("costMs", totalTime)

    getMemStat()

@ex.automain
def mainFunc():
    logger.info("starting speed test")
    for strategy in [StrategyRaw()]:
        for llm_model in [ModelQwenV15S14bChat]:
            device = None
            if llm_model == ModelQwenV15S14bChat:
                device = 0

            for framework in [FRAMEWORK.VLLM.name, FRAMEWORK.HUGGINGFACE.name]:
                for batch_size in [1, 2, 4]:
                    for use_cache in [True]:
                        for quantization_type in [None]:
                            for prompt_lookup_num_tokens in [None]:
                                if prompt_lookup_num_tokens and not use_cache:
                                    continue

                                experiment(
                                    strategy=strategy,
                                    llm_model=llm_model,
                                    framework=framework,
                                    device=device,
                                    batch_size=batch_size,
                                    use_cache=use_cache,
                                    quantization_type=quantization_type,
                                    prompt_lookup_num_tokens=prompt_lookup_num_tokens)

if __name__ == "__main__":
    pass