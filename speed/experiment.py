import time, logging
from sacred import Experiment
from milkie.agent.prompt_agent import PromptAgent
from milkie.config.config import FRAMEWORK, GlobalConfig
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory
from milkie.prompt.prompt import Loader
from milkie.settings import Settings
from milkie.strategy import Strategy, StrategyRaw

from milkie.utils.commons import getMemStat
from milkie.utils.data_utils import loadFromYaml

logger = logging.getLogger(__name__)

ModelYi34 = "/mnt/data1/.cache/huggingface/hub/01ai/Yi-34B-Chat/"

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelBaichuan13bChat = Prefix+"baichuan-inc/Baichuan2-13B-Chat"
ModelQwen14bChat = Prefix+"qwen/Qwen-14B-Chat"
ModelQwenV15S14bChat = Prefix+"qwen/Qwen1.5-14B-Chat/"


from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))
modelFactory = ModelFactory()

@ex.capture()
def experiment(
        strategy :Strategy,
        **kwargs):
    configYaml = loadFromYaml("config/global.yaml")
    if "llm_model" in kwargs:
        configYaml["llm"]["model"] = kwargs["llm_model"]

    if "framework" in kwargs:
        configYaml["llm"]["framework"] = kwargs["framework"]

    if "device" in kwargs:
        configYaml["llm"]["device"] = kwargs["device"]

    if "quantization_type" in kwargs:
        configYaml["llm"]["model_args"]["quantization_type"] = kwargs["quantization_type"]

    if "attn_implementation" in kwargs:
        configYaml["llm"]["model_args"]["attn_implementation"] = kwargs["attn_implementation"]
    
    if "torch_compile" in kwargs:
        configYaml["llm"]["model_args"]["torch_compile"] = kwargs["torch_compile"]

    if "repetition_penalty" in kwargs:
        configYaml["llm"]["generation_args"]["repetition_penalty"] = kwargs["repetition_penalty"]

    if "temperature" in kwargs:
        configYaml["llm"]["generation_args"]["temperature"] = kwargs["temperature"]

    if "do_sample" in kwargs:
        configYaml["llm"]["generation_args"]["do_sample"] = kwargs["do_sample"]
    
    if "use_cache" in kwargs:
        configYaml["llm"]["generation_args"]["use_cache"] = kwargs["use_cache"]
        
    if "prompt_lookup_num_tokens" in kwargs:
        configYaml["llm"]["generation_args"]["prompt_lookup_num_tokens"] = kwargs["prompt_lookup_num_tokens"]

    promptDict = Loader.loadByPrefix("speed/prompts", "prompt_")
    globalConfig = GlobalConfig(configYaml)

    globalConfig.memoryConfig = None
    globalContext = GlobalContext(globalConfig, modelFactory)
    context = Context(globalContext=globalContext)
    agent = strategy.createAgent(context)

    cnt = 0
    totalTime = 0
    totalTokens = 0
    for (_, prompts) in promptDict.items():
        for prompt in prompts:
            content = prompt.getContent()
            t0 = time.time()
            result = agent.task(content)
            totalTokens += result.metadata["numTokens"]
            t1 = time.time()
            logger.info(f"Testcase[{content[:5]}] Ans[{result}] cost[{t1-t0:.2f}]]")
            totalTime += t1-t0
            cnt += 1
    tokensPerSec = float(totalTokens)/totalTime

    #TODO: 1.412 is observered from the A800 GPU, need to remove this hard code
    logger.info(f"Running "
                f"kwargs[{kwargs}] "
                f"costSec[{totalTime}] "
                f"avgLatSec[{totalTime/cnt}] "
                f"tokensPerSec[{tokensPerSec}] "
                f"memory[{globalContext.settings.llm.getMem()}] "
                f"mbu[{globalContext.settings.llm.getMBU(tokensPerSec, 1.412 * 1024**4)}] ")
    ex.log_scalar("total", cnt)
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
                for use_cache in [True]:
                    for quantization_type in [None]:
                        for prompt_lookup_num_tokens in [None, 20]:
                            if prompt_lookup_num_tokens and not use_cache:
                                continue

                            experiment(
                                strategy=strategy,
                                llm_model=llm_model,
                                framework=framework,
                                device=device,
                                use_cache=use_cache,
                                quantization_type=quantization_type,
                                prompt_lookup_num_tokens=prompt_lookup_num_tokens)

if __name__ == "__main__":
    pass