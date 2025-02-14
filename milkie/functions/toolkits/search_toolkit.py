from typing import List
from milkie.functions.openai_function import OpenAIFunction
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.tools.basic_tools import ToolSearchWebFromDuckDuckGo, ToolGetWebContentFromUrls
from milkie.global_context import GlobalContext

class SearchToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

        self.toolWebSearch = ToolSearchWebFromDuckDuckGo()
        self.toolGetWebContentFromUrls = ToolGetWebContentFromUrls()

    def searchFromWeb(self, query: str) -> str:
        """
        从互联网搜索信息

        Args:
            query (str): 要搜索的查询字符串
        """
        return self.toolWebSearch.execute(query, maxResults=10)

    def getWebContentFromUrls(self, urls: str) -> str:
        """
        从给定url链接获取网页内容。

        Args:
            urls (str): 包含URL的文本
        """
        return self.toolGetWebContentFromUrls.execute(urls)

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.searchFromWeb),
            OpenAIFunction(self.getWebContentFromUrls)
        ]
