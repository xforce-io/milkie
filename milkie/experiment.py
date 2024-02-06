import logging
from sacred import Experiment
from milkie.config.config import GlobalConfig

from milkie.testsuite import TestCase, TestSuite
from milkie.utils.data_utils import loadFromYaml

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

@ex.config
def theConfig():
    reranker = "NONE"
    chunkSize = 256

@ex.capture()
def experiment(
        reranker, 
        chunk_size,
        channel_recall,
        similarity_top_k):
    configYaml = loadFromYaml("config/global.yaml")
    configYaml["index"]["chunk_size"] = chunk_size
    configYaml["retrieval"]["reranker"]["name"] = reranker
    configYaml["retrieval"]["channel_recall"] = channel_recall
    configYaml["retrieval"]["similarity_top_k"] = similarity_top_k

    globalConfig = GlobalConfig(configYaml)
    TestSuite("三体", TestCases).run(ex, globalConfig)

@ex.automain
def mainFunc():
    for reranker in ["FLAGEMBED"]:
        for chunkSize in [256]:
            for channel_recall in [30, 40]:
                for similarity_top_k in [20, 30]:
                    logging.info(f"reranker: {reranker}, chunkSize: {chunkSize}, channel_recall: {channel_recall}, similarity_top_k: {similarity_top_k}")
                    experiment(
                        reranker=reranker, 
                        chunk_size=chunkSize,
                        channel_recall=channel_recall,
                        similarity_top_k=similarity_top_k)

if __name__ == "__main__":
    pass