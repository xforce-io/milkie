from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.response import Response

class PaperReader(BaseAgent):
    
    def __init__(self, context: Context = None, config: str | GlobalConfig = None) -> None:
        super().__init__(context, config)

        flowCode = """
                1. 获取 {url} 页面内容 -> papersPage
                2. 页面内容如下：--{papersPage}--。从页面内容中提取出论文的标题和链接 => {{ 标题1:链接1, ... }} -> paperLinks
                FOR paperLink in paperLinks :
                    0. #PY ```"{url}"+"{paperLink.value}"``` -> fullLink
                    1. 获取 {fullLink} 页面内容 -> paperPage
                    2. 提取内容中arxiv pdf链接的相关内容，以及论文的发布日期，
                        链接模式为https://arxiv.org/pdf/xxxx.pdf
                        内容如下：----{paperPage}---- => {{link:论文链接, pubDate:发布日期}} -> paperInfo
                    3. #PY ```pubDate = "{paperInfo.pubDate}"; _NEXT_ if pubDate else _RET_```
                    4. #IF 如果今天距离{paperInfo.pubDate}天数绝对值不超过14天，则 返回 _NEXT_，否则返回 _RET_
                    5. 从{paperInfo.link}下载文件, 并且返回文件本地地址 -> paperLocalLink
                    6. 读取文件 {paperLocalLink} 中的内容 -> paperContent
                    7. 对下面的内容用中文进行用 markdown 格式总结，分为"问题概要"、"问题分析"、"问题解决"三部分阐述: {paperContent} -> paperDigest
                    8. #RET {{"title": "{{paperLink.key}}", "link": "{{fullLink}}", "digest": "{{paperDigest}}", "pubDate": "{{paperInfo.pubDate}}"}}
                END -> papers
                1. #PY ```from datetime import datetime; f"论文摘要-{source}-{{datetime.now().strftime('%Y-%m-%d')}}"``` -> paperTitleStr
                2. #PY ```
                    "\\n\\n".join([f"标题：{{paper['title']}}\\n链接：{{paper['link']}}\\n内容：{{paper['digest']}}\\n发布日期：{{paper['pubDate']}}\\n\\n" for paper in {papers}])
                ``` -> paperContentStr
                3. 根据下面信息发一份邮件
                    收件人:{emailAddr}
                    标题:[{paperTitleStr}]
                    正文:[{{paperContentStr}}]
            """

        self.flowBlock = FlowBlock(flowCode=flowCode, usePrevResult=False)
        self.flowBlock.compile()

    def execute(self, args: dict, **kwargs) -> Response:
        return self.flowBlock.execute(args=args)
    
if __name__ == "__main__":
    args = {
        "source" : "huggingface",
        "url" : "https://huggingface.co/papers",
        "emailAddr" : "freeman.xu@aishu.cn",
    }

    paperReader = PaperReader()
    paperReader.execute(args=args)
