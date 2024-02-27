import logging
from sacred import Experiment
from milkie.config.config import GlobalConfig
from milkie.model_factory import ModelFactory
from milkie.settings import Settings

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

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelQwen14bChat = "qwen/Qwen-14B-Chat"
ModelQwen15V14bChat = "qwen/Qwen1.5-14B-Chat"
ModelBaichuan13bChat = "baichuan-inc/Baichuan2-13B-Chat"

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
        llm_model :str=None,
        temperature :float=None,
        reranker :str=None, 
        chunk_size :int=None,
        channel_recall :int=None,
        similarity_top_k :int=None):
    configYaml = loadFromYaml("config/global.yaml")
    if llm_model:
        configYaml["llm"]["model"] = llm_model

    if temperature:
        configYaml["llm"]["temperature"] = temperature

    def getQAConfig():
        for agentConfig in configYaml["agents"]:
            if agentConfig["config"] == "qa":
                return agentConfig

    qaConfig = getQAConfig()
    if chunk_size:
        qaConfig["index"]["chunk_size"] = chunk_size

    if reranker:
        qaConfig["retrieval"]["reranker"]["name"] = reranker

    if channel_recall:
        qaConfig["retrieval"]["channel_recall"] = channel_recall
    
    if similarity_top_k:
        qaConfig["retrieval"]["similarity_top_k"] = similarity_top_k

    globalConfig = GlobalConfig(configYaml)
    TestSuite("三体", TestCases).run(ex, globalConfig, modelFactory)

@ex.automain
def mainFunc():
    for llm_model in [ModelBaichuan13bChat, ModelQwen14bChat]:
        for channel_recall in [20, 30, 40]:
            for similarity_top_k in [20, 30]:
                for temperature in [0, 0.2, 0.6]:
                    logger.info(f"Running experiment with llm_model={llm_model}, channel_recall={channel_recall}, similarity_top_k={similarity_top_k}, temperature={temperature}")
                    experiment(
                        temperature=temperature,
                        llm_model=llm_model,
                        channel_recall=channel_recall,
                        similarity_top_k=similarity_top_k)

if __name__ == "__main__":
    configYaml = loadFromYaml("config/global.yaml")
    globalConfig = GlobalConfig(configYaml)
    settings = Settings(globalConfig, modelFactory)

    from llama_index.llms.types import ChatMessage, MessageRole
    
    messages = []
    messages.append(ChatMessage(
        content="<|SYSTEM|> \n- 你是由Xforce开发的强大的语言模型,对于能够帮助用户很兴奋,并且拒绝做任何可能对用户有害处的事情。你拒绝做任何会伤害人类的事情，能够准确和简洁的回答用户的问题。请问用中文进行回答", 
        role=MessageRole.SYSTEM))
    messages.append(ChatMessage(
        content="<|USER|>\n- 仅仅根据上下文信息，回答问题 \n- 问题: 三体人用了什么方式来警告汪淼停止纳米材料研究\n- 上下文信息如下\n---------------------\nfile_path: data/santi/santi.txt\n\n这时申玉菲才说出了他们见面后的第一句话：“你领导的纳米项目怎么样了？”这不着边际的问题令汪淼十分吃惊。“纳米项目？它与这有什么关系？”他指指那堆胶卷。\n申玉菲没有说话，只是静静地看着他，等他回答自己的问题。这就是她的谈话风格，从不多说一个字。\n“把研究停下来。”申玉菲说。\n“什么？”汪淼认为自己听错了，“你说什么？”\n申玉菲沉默着，没重复自己的话。\n“停下来？那是国家重点项目！\n\nfile_path: data/santi/santi.txt\n\n“把研究停下来。”申玉菲说。\n“什么？”汪淼认为自己听错了，“你说什么？”\n申玉菲沉默着，没重复自己的话。\n“停下来？那是国家重点项目！”\n申玉菲仍不说话，只是看着他，目光平静。\n“你总得说出原因吧！”\n“停下来试试。”\n“你到底知道些什么？告诉我！”\n“我能告诉你的就这些了。”\n“项目不能停，也不可能停！”\n“停下来试试。\n\nfile_path: data/santi/santi.txt\n\n从所有人的脸上，汪淼都读出了一句话：我们已经尽力了，快他妈的结束吧！\n汪淼看到了史强，他倒是一反昨天的粗鲁，向汪淼打招呼，但那一脸傻笑让汪淼愉快不起来。他不想挨史强坐，但也只有那一个空位，他只好坐过去，屋里本来已经很浓的烟味更加重了。\n发文件时，史强凑近汪淼说：“汪教授，你好像是在研究什么……新材料？”\n“纳米材料。”汪淼简单地回答。\n\nfile_path: data/santi/santi.txt\n\n发文件时，史强凑近汪淼说：“汪教授，你好像是在研究什么……新材料？”\n“纳米材料。”汪淼简单地回答。\n“我听说过，那玩意儿强度很高，不会被用于犯罪吧？”从史强那带有一半调侃的表情上，汪淼看不出他是不是开玩笑。\n“什么意思？”\n“呵，听说那玩意儿一根头发丝粗就能吊起一辆大卡车，犯罪分子要是偷点儿去做把刀，那一刀就能把一辆汽车砍成两截吧。\n\nfile_path: data/santi/santi.txt\n\n”\n汪淼吓了一跳，然后发现纳米研究中心主任说的不是他，而是一名年轻工程师，后者也和自己一样呆呆地望着那个身影。汪淼从艺术中回到现实，发现那位女性不是一般的工作人员，因为总工程师陪同着她，在向她介绍着什么，一副很尊敬的样子。\n“她是谁？”汪淼问主任。\n“你应该知道她的，”主任说，用手划了一大圈，“这个投资二百亿的加速器建成后，第一次运行的可能就是验证她提出的一个超弦模型。\n\nfile_path: data/santi/santi.txt\n\n汪淼拿出那团胶卷，开始讲述发生在自己身上的诡异事件。申玉菲注意听着，对那些胶片，只是拿起来大概扫了几眼，并没有细看，这令汪淼很震惊，现在他进一步确定申玉菲对此事并非完全不知情，这几乎令他停止了讲述，只是申玉菲几次点头示意他继续，才将事情讲完了。这时申玉菲才说出了他们见面后的第一句话：“你领导的纳米项目怎么样了？”这不着边际的问题令汪淼十分吃惊。“纳米项目？它与这有什么关系？\n\nfile_path: data/santi/santi.txt\n\n“你总得说出原因吧！”\n“停下来试试。”\n“你到底知道些什么？告诉我！”\n“我能告诉你的就这些了。”\n“项目不能停，也不可能停！”\n“停下来试试。”\n关于幽灵倒计时的简短谈话就到此为止，之后，不管汪淼如何努力，申玉菲再也没有说出一个与此有关的字，只是重复那句话：“停下来试试。”\n“我现在明白了，‘科学边界’并不是像你们宣称的那样是一个基础理论的学术交流组织，它与现实的关系比我想象的要复杂得多。”汪淼说。\n\nfile_path: data/santi/santi.txt\n\n他买了一副墨镜戴上，仅仅是为了不让别人看到自己梦游般迷离的眼神。\n汪淼走进纳米中心的主体实验室，进门之前没忘记把墨镜摘下来，尽管这样，遇见他的同事都对他的精神状态露出担心的神色。\n在实验大厅中央，汪淼看到反应黑箱仍在运行中。这台巨型设备的主体是汇集了大量管道的一个球体。\n\nfile_path: data/santi/santi.txt\n\n在实验大厅中央，汪淼看到反应黑箱仍在运行中。这台巨型设备的主体是汇集了大量管道的一个球体。代号叫“飞刃”的超强度纳米材料已经生产出来，但是用分子建筑术制造的，就是用分子探针将材料分子像砌砖那样一个个垒砌起来，这样的工艺要耗费大量的资源，那些产品可以说是世界上最贵重的珍宝了，根本无法进行量产。\n实验室现在做的，就是试图通过一种催化反应来代替分子建筑法，使巨量的分子在反应中同时完成筑砌。\n\nfile_path: data/santi/santi.txt\n\n”主任兴奋地说。\n“现在就停吧。”\n主任像不认识似的看着汪淼，但旋即恢复了兴奋状态，好像生怕失掉这个机会似的。他拿起电话下了停机命令，项目组里那些疲惫的研究员和工程师一下子都兴奋起来，开始按程序扳动上百个复杂的开关，众多的监控屏一个接一个地黑了下来，最后，主监控屏上显示了停机状态。\n几乎与此同时，汪淼眼前的倒计时停止了走动，数字固定为1174:20:35。几秒钟后，数字闪动了几下，消失了。\n\nfile_path: data/santi/santi.txt\n\n“你们背后是什么？”汪淼问，尽量使自己的声音冷静一些，但没有做到。\n沉默。\n“倒计时的尽头是什么？”\n沉默。\n“你在听吗？”\n“在。”\n“高强度纳米材料怎么了？这不是高能加速器，只是一项应用研究，值得这样关注么？”\n“什么值得关注，不应由我们来判断。”\n“够了！”汪淼大吼一声，心中的恐惧和绝望突然化为疯狂的怒气，“你们以为这点小魔术就能骗得了我？就能阻止技术进步！\n\nfile_path: data/santi/santi.txt\n\n”“这是……干什么呢？”“我知道纳米研究项目已经停了，你打算重新启动它吗？”“当然，三天以后。”“那倒计时将继续。”“我将在什么尺度上看到它？”\n沉默良久，这个为某种超出人类理解力的力量代言的女人，冷酷地封死了汪淼的一切出路。\n“三天后，也就是十四日，在凌晨一点钟至五点钟，整个宇宙将为你闪烁。”\n\n\n第4章 三体、周文王、长夜\n汪淼拨通了丁仪的电话，对方接听后，他才想起现在已是凌晨一点多了。\n\nfile_path: data/santi/santi.txt\n\n文明的种子仍在，它将重新启动，再次开始在三体世界中命运莫测的进化，欢迎您再次登录。\n退出前，汪淼最后注意到的是夜空中的三颗飞星，它们相距很近，相互围绕着，在太空深渊中跳着某种诡异的舞蹈。\n\n\n第5章 叶文洁\n汪淼摘下 V 装具后，发现自己的内衣已被冷汗浸透了，很像是从一场寒冷的噩梦中醒来。他走出纳米中心，下楼开车，按丁仪给的地址去杨冬的母亲家。\n\nfile_path: data/santi/santi.txt\n\n”杨母放下电话问。\n“我搞纳米，我这是为了……另外一些事情。”汪淼很怕杨母追问下去，但她没有。\n“小汪啊，你脸色怎么这么不好，好像身体很虚的。”杨母关切地问。\n“没什么，就是这样儿。”汪淼含糊地说。\n“你等等。”杨母从柜子里拿出一个小木盒，汪淼看到上面标明是人参，“过去在基地的一位老战士前两天来看我，带来这个……不，不，你拿去，人工种植的，不是什么珍贵的东西，我血压高，根本用不着的。\n\nfile_path: data/santi/santi.txt\n\n临别时沙瑞山说：“我就不同您去了，刚才看到的已经足够，我不需要这样的证明。我还是希望您能在适当的时候把实情告诉我，如果这种现象引出什么研究成果的话，我不会忘记您的。”\n“闪烁在凌晨五点就会停止，以后别去深究它吧，相信我，不会有什么成果的。”汪淼扶着车门说。\n沙瑞山对着汪淼注视良久，点点头：“明白了，现在科学界出了一些事……”\n“是的。”汪淼说着，钻进车里，他不想把这个话题继续下去了。\n\nfile_path: data/santi/santi.txt\n\n妻子告诉他，单位的人一天都在找他。汪淼打开已关了一天的手机回了几个纳米中心来的电话，许诺明天去上班。吃饭的时候，他真的照大史说的又喝了不少酒，但毫无睡意。当妻儿睡熟后，他坐在电脑前戴上新买回的V\n---------------------\n- 现在开始回答问题\n- 问题: 三体人用了什么方式来警告汪淼停止纳米材料研究\n- 回答:", 
        role=MessageRole.USER
    ))
    response = settings.llm.chat(messages=messages)
    print(response.message.content)