import logging
from sacred import Experiment
from milkie.config.config import FRAMEWORK, GlobalConfig
from milkie.model_factory import ModelFactory
from milkie.strategy import Strategy, StrategyDeepQA

from milkie.benchmark.testsuite import TestCase, TestSuite
from milkie.utils.data_utils import loadFromYaml
from playground.model_repos import GModelRepo

logger = logging.getLogger(__name__)


TestCases = [
    TestCase("三体人用了什么方式来警告汪淼停止纳米材料研究", [["幽灵", "宇宙背景辐射"], "倒计时"]),
    TestCase("三体世界的纪元主要分哪几种", ["恒纪元", "乱纪元"]),
    TestCase("三体人有哪种独特的生存形态", ["脱水"]),
    TestCase("三体人如何繁殖后代", ["分裂"]),
    TestCase("三体世界最恐怖的灾难是什么", [["三日凌空", "三颗太阳"]]),
    TestCase("三体人最早的计算机是用什么组成硬件的", ["人列计算机"]),
    TestCase("1379号观察员给地球发送的警告的主要内容", ["不要回答"]),
    TestCase("ETO内部主要分为哪几派", ["降临派", "拯救派", "幸存派"]),
    TestCase("叶文洁的爷爷的儿子叫什么名字", ["叶哲泰"]),
    TestCase("切割“审判日号”的行动代号是什么", ["古筝"]),
    TestCase("叶文雪的爸爸是谁", ["叶哲泰"]),
    TestCase("汪淼第一次玩三体游戏时，那个世界演化到了哪个阶段，毁于寒冷还是炙热", ["战国", ["寒冷", "严寒"]]),
    TestCase("红岸系统的目标是摧毁敌人的卫星吗", ["不是"]),
    TestCase("申玉菲和他丈夫初次见面是在哪里", ["寺庙"]),
    TestCase("在三体游戏中，是冯诺依曼和爱因斯坦共同为秦始皇设计了人列计算机，对不对", ["不对"]),
    TestCase("云河咖啡馆聚会中，主持人是谁", ["潘寒"]),
    TestCase("谁认为欧洲人入侵南美洲是强盗，哲学家、软件公司副总、女作家、电网领导还是博士生", ["软件公司副总", "电网领导"]),
    TestCase("杨冬的外婆叫什么名字", ["绍琳"]),
    TestCase("第二红岸基地的建造经费主要来自于谁", ["伊文斯"]),
    TestCase("杀死申玉菲的凶手希望地球叛军统帅属于哪个派别", ["降临派"]),
]

from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver("my_runs"))
modelFactory = ModelFactory()


@ex.config
def theConfig():
    reranker = "FLAGEMBED"
    chunkSize = 256

@ex.capture()
def experiment(
        strategy :Strategy,
        **kwargs):
    configYaml = loadFromYaml("config/global.yaml")
    if "llm_model" in kwargs:
        configYaml["llm"]["model"] = kwargs["llm_model"]

    if "load_in_8bit" in kwargs:
        configYaml["llm"]["model_args"]["load_in_8bit"] = kwargs["load_in_8bit"]

    if "attn_implementation" in kwargs:
        configYaml["llm"]["model_args"]["attn_implementation"] = kwargs["attn_implementation"]

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

    def getAgentConfig(name :str):
        for agentConfig in configYaml["agents"]:
            if agentConfig["config"] == name:
                return agentConfig

    agentConfig = getAgentConfig(strategy.getAgentName())
    if "reranker" in kwargs:
        agentConfig["retrieval"]["reranker"]["name"] = kwargs["reranker"]

    if "rerank_position" in kwargs:
        agentConfig["retrieval"]["reranker"]["position"] = kwargs["rerank_position"]

    if "rewrite_strategy" in kwargs:
        agentConfig["retrieval"]["rewrite_strategy"] = kwargs["rewrite_strategy"]

    if "chunk_size" in kwargs:
        agentConfig["index"]["chunk_size"] = kwargs["chunk_size"]

    if "channel_recall" in kwargs:
        agentConfig["retrieval"]["channel_recall"] = kwargs["channel_recall"]
    
    if "similarity_top_k" in kwargs:
        agentConfig["retrieval"]["similarity_top_k"] = kwargs["similarity_top_k"]

    globalConfig = GlobalConfig(configYaml)
    TestSuite("三体", TestCases).run(
        strategy, 
        ex, 
        globalConfig, 
        modelFactory, 
        **kwargs)

@ex.automain
def mainFunc(
        strategy,
        llm_model,
        framework,
        rewrite_strategy):
    kwargs = {
        "strategy":Strategy.getStrategy(strategy),
        "llm_model":GModelRepo.getModel(llm_model).getModelPath(),
        "framework":framework,
        "rewrite_strategy":rewrite_strategy,
        "reranker":"FLAGEMBED",
        "rerank_position":"NONE",
        "channel_recall":30,
        "similarity_top_k":20
    }
    experiment(**kwargs)