import time, logging
from sacred import Experiment

from llama_index.legacy.response.schema import Response

from milkie.benchmark.benchtype import BenchTypeKeyword, Benchmarks
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.prompt.prompt import Loader
from milkie.strategy import Strategy
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

PrefixAishuReader = "/mnt/data2/.cache/huggingface/hub/"
ModelAishuReader2Chat = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2"

from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))

def getModel(name :str) -> str:
    if name == "Yi34":
        return ModelYi34
    elif name == "Baichuan13bChat":
        return ModelBaichuan13bChat
    elif name == "Qwen14bChat":
        return ModelQwen14bChat
    elif name == "QwenV15S14bChat":
        return ModelQwenV15S14bChat
    elif name == "QwenV15S14bGPTQINT4Chat":
        return ModelQwenV15S14bGPTQINT4Chat
    elif name == "QwenV15S14bGPTQINT8Chat":
        return ModelQwenV15S14bGPTQINT8Chat
    elif name == "QwenV15S14bAWQChat":
        return ModelQwenV15S14bAWQChat
    elif name == "AishuReader2Chat":
        return ModelAishuReader2Chat
    else:
        raise ValueError(f"Unknown model name: {name}")

@ex.config
def theConfig():
    strategy = "raw"
    llm_model = "QwenV15S14bChat"
    framework = "LMDEPLOY"
    batch_size = 10
    use_cache = True
    quantization_type = None
    prompt_lookup_num_tokens = None
    prompt = "qa_strict"
    benchmarks = ""

@ex.capture()
def experiment(
        strategy :Strategy,
        **kwargs):
    promptQA = Loader.load(kwargs["prompt"])

    globalConfig = makeGlobalConfig(**kwargs)
    globalConfig.memoryConfig = None

    modelFactory = ModelFactory()
    globalContext = GlobalContext(globalConfig, modelFactory)
    context = Context(globalContext=globalContext)
    agent = strategy.createAgent(context)

    benchmarks = Benchmarks(
        [BenchTypeKeyword(benchmark.strip()) for benchmark in kwargs["benchmarks"].split(";")],
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
def mainFunc(
        strategy, 
        llm_model, 
        framework, 
        batch_size, 
        use_cache, 
        quantization_type, 
        prompt_lookup_num_tokens,
        benchmarks):
    logger.info("starting speed test")

    assert type(batch_size) == int
    assert type(use_cache) == bool
    assert not (prompt_lookup_num_tokens and not use_cache)

    kwargs = {
        "strategy":Strategy.getStrategy(strategy),
        "llm_model":getModel(llm_model),
        "framework":framework,
        "batch_size":batch_size,
        "use_cache":use_cache,
        "quantization_type":quantization_type,
        "prompt_lookup_num_tokens":prompt_lookup_num_tokens,
        "benchmarks":benchmarks,
    }

    experiment(**kwargs)

if __name__ == "__main__":
    pass