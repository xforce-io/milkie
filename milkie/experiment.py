import logging
from sacred import Experiment
from milkie.config.config import GlobalConfig
from milkie.model_factory import ModelFactory
from milkie.settings import Settings
from milkie.strategy import Strategy, StrategyDeepQA, StrategyMRQA

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

ModelYi34 = "/mnt/data1/.cache/huggingface/hub/01ai/Yi-34B-Chat/"

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelBaichuan13bChat = Prefix+"baichuan-inc/Baichuan2-13B-Chat"
ModelQwen14bChat = Prefix+"qwen/Qwen-14B-Chat"
ModelQwenV15S14bChat = Prefix+"qwen/Qwen1.5-14B-Chat/"

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

    if "temperature" in kwargs:
        configYaml["llm"]["temperature"] = kwargs["temperature"]

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
def mainFunc():
    for strategy in [StrategyDeepQA()]:
        for llm_model in [ModelQwen14bChat, ModelQwenV15S14bChat]:
            for reranker in ["NONE", "FLAGEMBED"]:
                for rerank_position in ["NONE", "SIMPLE"]:
                    for rewrite_strategy in ["NONE", "QUERY_REWRITE"]:
                        for channel_recall in [30]:
                            for similarity_top_k in [20, 40]:
                                experiment(
                                    strategy=strategy,
                                    llm_model=llm_model,
                                    reranker=reranker,
                                    rerank_position=rerank_position,
                                    rewrite_strategy=rewrite_strategy,
                                    channel_recall=channel_recall,
                                    similarity_top_k=similarity_top_k)

if __name__ == "__main__":
    configYaml = loadFromYaml("config/global.yaml")
    globalConfig = GlobalConfig(configYaml)
    settings = Settings(globalConfig, modelFactory)

    from llama_index.llms.types import ChatMessage, MessageRole
    
    messages = []
    messages.append(ChatMessage(
        content="- 你是由Xforce开发的强大的语言模型,对于能够帮助用户很兴奋,并且拒绝做任何可能对用户有害处的事情。你拒绝做任何会伤害人类的事情，能够准确和简洁的回答用户的问题。请问用中文进行回答", 
        role=MessageRole.SYSTEM))
    messages.append(ChatMessage(
        content="- 仅仅根据上下文信息，回答问题 //- 问题: 三体人如何繁殖后代//- 上下文信息如下//---------------------//file_path: data/santi/santi.txt////后来我们结婚了，没多少爱情和激情，只是为了双方生活的方便而已，我们都有各自的事情要做。对我来说，以后的几年可以用一天来形容，日子在平静中就过去了。在那幢别墅里，我衣来伸手饭来张口，只需专注于三体问题的研究就行了。申玉菲从不干涉我的生活，车库里有我的一辆车，我可以开着它去任何地方，我甚至敢肯定，自己带一个女人回家她都不在乎，她只关注我的研究。我们每天唯一交流的内容就是三体问题，她每天都要了解研究的进展。////file_path: data/santi/santi.txt////后来人们都热衷于寻找这种特殊的稳定状态，找到一个就乐得跟什么似的，到目前为止也就是找到了三四种。其实，我用进化算法已经找到了一百多种稳定状态，把那些轨迹画出来，足够办一个后现代派画展了。但这不是我的目标，三体问题的真正解决，是建立这样一种数学模型，使得三体在任何一个时间断面的初始运动矢量已知时，能够精确预测三体系统以后的所有运动状态。这也是申玉菲渴望的目标。//但平静的生活到昨天就结束了，我遇到了麻烦事。//“这就是你要报的案了吧？////file_path: data/santi/santi.txt////你们刚才问过三体人的外形，据一些迹象推测，构成人列计算机的三体人，外表可能覆盖着一层全反射镜面，这种镜面可能是为了在恶劣的日照条件下生存而进化出来的，镜面可以变化出各种形状，他们之间就通过镜面聚焦的光线来交流，这种光线语言信息传输的速度是很快的，这就是人列计算机得以存在的基础。当然，这仍是一台效率很低的机器，但确实能够完成人类手工力不能及的运算。计算机在三体世界首先确实是以人列形式出现，然后才是机械式和电子式的。////file_path: data/santi/santi.txt////据学者们的预测，最近的一次膨胀将在一百五十至二百万年后发生。”//“这个鬼地方，实在是待不下去了”爱因斯坦用一个老乞丐的姿势抱着小提琴蹲在地上说。//秘书长点点头说：“待不下去了，也不能再待下去了！三体文明的唯一出路，就是和这个宇宙赌一把。”//“怎么赌？”汪淼问。//“飞出三体星系，飞向广阔的星海，在银河系中寻找可以移民的新世界！////file_path: data/santi/santi.txt////如果在五千个三体时之内还找不到工作，他也将面临着强制脱水后被焚烧掉的命运。//逃脱这种命运的唯一途径是与一名异性组合。这时，构成他们身体的有机物质将融为一体，其中三分之二的物质将成为生化反应的能源，使剩下的三分之一细胞完成彻底的更新，生成一个全新的躯体；之后这个躯体将发生分裂，裂解为三至五个新的幼小生命，这就是他们的孩子，他们将继承父母的部分记忆，成为他们生命的延续，重新开始新的人生。////file_path: data/santi/santi.txt////如果是后者，那么他就有了使自己这卑微的生命燃烧一次的机会。//同地球的红岸基地一样，三体世界的大部分监听站也在同时向太空中发射信息，呼唤可能存在的外星文明。三体科学家也早就发现了恒星对于电波的放大功能，遗憾的是半人马区的三颗太阳在结构上与人类的太阳有很大差异，存在着很大的外围等离子气层(正是这个气层使三体世界的太阳在一定的距离上突然变成飞星或由飞星显形)，这种气层对电磁波有很强的屏蔽作用，使得到达太阳能量镜面的电波功率有一个极大的阙值，因而不可能把太阳作为天线发射信息，只能用地面天线直接向目标发射。////file_path: data/santi/santi.txt////”//“对，消灭地球文明还有另外一个理由：他们也是好战的种族，很危险。当我们与其共存于一个世界时，他们在技术上将学得很快，这样下去，两个文明都过不好。我们已经确定的政策是：三体舰队占领太阳系和地球后，不会对地球文明进行太多干涉，地球人完全可以像以前那样生活，就像三体占领者不存在一样，只有一件事是被永远禁止的：生育。现在我要问：你想当地球的救世主，对自己的文明却没有一点责任感？”//“三体世界已经让我厌倦了。////file_path: data/santi/santi.txt////第二个计划代号‘神迹’。即对地球人进行的超自然力量的展示，这个计划力图通过一系列的‘神迹’，建造一个科学逻辑无法解释的虚假宇宙。当这种假象持续一定时间后，将有可能使三体文明在那个世界成为宗教信徒的崇拜对象，在地球的思想界，非科学的思维方式就会压倒科学思维，进而导致整个科学思想体系的崩溃。”//“如何产生神迹呢？”//“神迹之所以成为神迹，关键在于它是地球人绝对无法识破的。////file_path: data/santi/santi.txt////”//“这事情不会发生，我保证！你干吗总跟我过不去？我说过，科学实验嘛……”//“够了！”元首说，“下次的成功率有多大？”//“几乎是百分之百！元首，请相信我，通过这两次失败我们已经掌握了微观至宏观低维展开的规律。”//“好吧，为了三体文明的生存，这个险必须冒。”//“谢谢元首！”//“但，如果下次还是失败，你，还有参与智子工程的所有科学家，都有罪了。”//“是的，当然，都有罪。”如果三体人能出汗的话，科学执政官一定抹了一把冷汗。////file_path: data/santi/santi.txt////“我们计算过，每个智子可以控制多达一万台次高能加速器，而地球人建造一台这样的加速器就需要四五年的时间，从经济和资源的角度看也不可能大量建造。当然，他们可以拉大加速器间的距离，比如说在他们星系的各个行星上建造，这确实能破坏智子的多线程操作，但在这样长的时间内，三体世界再造出十个或更多的智子也不困难。////file_path: data/santi/santi.txt////当然，他们可以拉大加速器间的距离，比如说在他们星系的各个行星上建造，这确实能破坏智子的多线程操作，但在这样长的时间内，三体世界再造出十个或更多的智子也不困难。越来越多的智子将在那个行星系中游荡，它们合在一起也没有细菌的亿万分之一那么大，但却使地球上的物理学家们永远无法窥见物质深处的秘密，地球人对微观维度的控制，将被限制在五维以下，别说是四百五十万时，就是四百五十万亿时，地球文明的科学技术也不会有本质的突破，它们将永远处于原始时代。////file_path: data/santi/santi.txt////这些‘神迹’将制造一种足以将人类科学思想引上歧途的氛围，这样，我们可以用神迹计划对地球世界中物理学以外的科学形成强有力的遏制。”//“最后一个问题：为什么不把已有的四个智子全部发往地球呢？”//“量子感应是超距的，即使四个智子分处宇宙的两端，感应照样可以在瞬间传递，它们构成的量子阵列依然存在。把三号和四号智子留在这里，它们就可以实时接收位于地球的一号和三号智子发回的信息，这样就实现了三体世界对地球的实时监视。////file_path: data/santi/santi.txt////把三号和四号智子留在这里，它们就可以实时接收位于地球的一号和三号智子发回的信息，这样就实现了三体世界对地球的实时监视。同时，智子阵列也使三体世界能够与地球文明中的异己分子进行实时通讯。”//“这里有一个重要的战略步骤，”元首插话说，“我们将通过智子阵列，把三体世界对地球文明的真实意图告诉地球人。”//“这就是说，我们将告诉他们，三体舰队将通过长期禁止地球人生育，使这个物种从地球上消失？////file_path: data/santi/santi.txt////”//“这就是说，我们将告诉他们，三体舰队将通过长期禁止地球人生育，使这个物种从地球上消失？”//“是的，这样做有两个可能的结果：其一是使地球人抛弃一切幻想决一死战，其二是他们的社会在绝望和恐惧之中堕落、崩溃。通过对已经收到的地球文明信息进行仔细研究，我们认为后一种可能性更大。”//不知什么时候，初升的太阳又消失在地平线下，日出变成了日落，三体世界的又一个乱纪元开始了。////file_path: data/santi/santi.txt////其二：星舰文明选择，即逃亡的人类把飞船作为永久居住地，使人类文明在永远的航行中延续。这个选择面临着与新世界选择相同的困难，只是更多偏重于建立小型自循环生态系统的技术，这种世代运行的全封闭生态圈远远超出了人类目前的技术能力。其三：暂避选择，在三体文明已经在太阳系完成定居后，已经逃亡到外太空的人类与三体社会积极交流，等待和促成其对外太空残余人类政策的缓和，最后重返太阳系，以较小的规模与三体文明共同生存。////file_path://---------------------//- 现在开始回答问题//- 问题: 三体人如何繁殖后代//- 回答: ", 
        role=MessageRole.USER
    ))
    response = settings.llm.chat(messages=messages)
    print(f"response[{response.message.content}]")
