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

###############################MODEL REPOS########################################
###############################MODEL REPOS########################################

ModelYi34 = "/mnt/data1/.cache/huggingface/hub/01ai/Yi-34B-Chat/"

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelBaichuan13bChat = Prefix+"baichuan-inc/Baichuan2-13B-Chat"
ModelQwen14bChat = Prefix+"qwen/Qwen-14B-Chat"
ModelQwenV15S14bChat = Prefix+"qwen/Qwen1.5-14B-Chat/"
ModelQwenV15S14bGPTQINT4Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int4/"
ModelQwenV15S14bGPTQINT8Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int8/"
ModelQwenV15S14bAWQChat = Prefix+"qwen/Qwen1___5-14B-Chat-AWQ/"

PrefixAishuReader = "/mnt/data2/.cache/huggingface/hub/"
ModelAishuReader2_Chat = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2"
ModelAishuReader2_Chat_AWQ = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2-awq"
ModelAishuReader2_Chat_GPTQ8 = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2-gptq-int8"

PrefixInternlm2 = "/mnt/data3/models/"
ModelInternlm2_Chat_20b = PrefixInternlm2 + "internlm2-chat-20b_v2"

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
    elif name == "AishuReader2_Chat":
        return ModelAishuReader2_Chat
    elif name == "AishuReader2_Chat_AWQ":
        return ModelAishuReader2_Chat_AWQ
    elif name == "AishuReader2_Chat_GPTQ8":
        return ModelAishuReader2_Chat_GPTQ8
    elif name == "Internlm2_Chat_20b":
        return ModelInternlm2_Chat_20b
    else:
        raise ValueError(f"Unknown model name: {name}")

###############################MODEL REPOS########################################
###############################MODEL REPOS########################################

from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))

@ex.config
def theConfig():
    strategy = "raw"
    llm_model = "QwenV15S14bChat"
    framework = "LMDEPLOY"
    batch_size = 50
    prompt_lookup_num_tokens = None
    system_prompt = None
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

    benchmarks = []
    for benchmark in kwargs["benchmarks"].split(";"):
        if benchmark.strip().empty():
            continue
        benchmarks.append(BenchTypeKeyword(benchmark.strip()))

    benchmarks = Benchmarks(
        ex,
        benchmarks,
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
    ex.log_scalar("total", numQueries)
    ex.log_scalar("costSec", totalTime)
    ex.log_scalar("avgQueryLatSec", totalTime/numQueries)
    ex.log_scalar("avgBatchLatSec", totalTime/numBatches)
    ex.log_scalar("tokensPerSec", tokensPerSec)
    ex.log_scalar("memory", globalContext.settings.llm.getMem())
    ex.log_scalar("mbu", globalContext.settings.llm.getMBU(tokensPerSec, 1.412 * 1024**4))

    getMemStat()

@ex.main
def mainFunc(
        strategy, 
        llm_model, 
        framework, 
        batch_size, 
        prompt_lookup_num_tokens,
        system_prompt,
        prompt,
        benchmarks):
    assert type(batch_size) == int
    kwargs = {
        "strategy":Strategy.getStrategy(strategy),
        "llm_model":getModel(llm_model),
        "framework":framework,
        "batch_size":batch_size,
        "prompt_lookup_num_tokens":prompt_lookup_num_tokens,
        "system_prompt":system_prompt,
        "prompt":prompt,
        "benchmarks":benchmarks,
    }

    experiment(**kwargs)

if __name__ == "__main__":
    result = ex.run()

    outputData = {
        "config" : result.config,
        "metrics" : result.to_json()
    }
    logger.info(f"exp result: {outputData}")