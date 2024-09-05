import logging
from datetime import datetime
from urllib.parse import urlparse, urljoin
from llama_index.core import Response
from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.llm_block import LLMBlock
from milkie.log import INFO

logger = logging.getLogger(__name__)

class PaperAgent(BaseAgent):

    def __init__(self, url: str) -> None:
        super().__init__(context=None, config=None)
        self.url = url
        self.domain = urlparse(url).netloc
        self._llmBlock = LLMBlock(context=self.context, config=self.config)

    def execute(self, emailAddr: str, **kwargs) -> Response:
        paperLinks = self._getPaperLinks()
        emailContent = self._processPapers(paperLinks)
        if len(emailContent) == 0:
            INFO(logger,f"没有找到任何论文")
            return Response(response="没有找到任何论文", metadata={})

        return self._generateEmail(emailContent, emailAddr)

    def _getPaperLinks(self) -> list[tuple[str, str]]:
        response = self._llmBlock.execute(f"""
            1. 获取 {self.url} 页面内容 -> papersPage
            2. 页面内容如下：--{{papersPage}}--。从页面内容中提取出所有论文的标题和链接 => {{{{ 标题1:链接1, ... }}}} -> paperLinks
        """)
        return [(title, link) for title, link in response.metadata["paperLinks"].items()]

    def _getPaperContent(self, paperInfoLink: str) -> dict:
        response = self._llmBlock.execute(f"""
            1. 获取 {paperInfoLink} 页面内容 -> paperPage
            2. 提取内容中arxiv pdf链接的相关内容，以及论文的发布日期，
                链接模式为https://arxiv.org/pdf/xxxx.pdf
                内容如下：----{{paperPage}}---- => {{{{link:论文链接, pubDate:发布日期}}}} -> paperInfo
            3. #PY ```pubDate = "{{paperInfo.pubDate}}"; _NEXT_ if pubDate else _END_```
            4. #IF 如果今天距离{{paperInfo.pubDate}}天数绝对值不超过八天，则 返回 _NEXT_，否则返回 _END_
            5. 从{{paperInfo.link}}下载文件, 并且返回文件本地地址 -> paperLocalLink
            6. 读取文件 {{paperLocalLink}} 中的内容 -> paperContent
            7. 对下面的内容用中文进行用 markdown 格式总结，分为"问题概要"、"问题分析"、"问题解决"三部分阐述: {{paperContent}}
        """)
        return {
            "content": response.response.response,
            "metadata": response.metadata
        }

    def _generateEmail(self, emailContent: list[dict[str, str]], emailAddr: str) -> Response:
        paperTitleStr = f"论文摘要-{self.domain}-{datetime.now().strftime('%Y-%m-%d')}"
        paperContentStr = "\n\n".join(
            f"标题：{content['title']}\n链接：{content['link']}\n内容：{content['content']}\n发布日期：{content['publishDate']}"
            for content in emailContent)
        self._llmBlock.setVarDict("paperContentStr", paperContentStr)
        return self._llmBlock.execute(f"""
            使用邮件工具发一份电子邮件给{emailAddr}
            标题:[[{paperTitleStr}]]
            正文:[[#paperContentStr#]]
        """, decomposeTask=False).response.response

    def _processPapers(self, paperLinks: list[tuple[str, str]]) -> list[dict[str, str]]:
        paperContents = []
        for title, link in paperLinks:
            paperLink = urljoin(f"https://{self.domain}", link)
            paperContent = self._getPaperContent(paperLink)
            if "问题概要" not in paperContent["content"]: continue

            paperContents.append({
                "title": title,
                "link": paperLink,
                "content": paperContent["content"],
                "publishDate": paperContent["metadata"]["paperInfo"]["pubDate"],
            })
        return paperContents

if __name__ == "__main__":
    print(PaperAgent("https://paperswithcode.com/").execute("freeman.xu@aishu.cn"))