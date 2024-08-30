import json
import re
import requests
from newspaper import Article
from typing import Any, Dict, List

from milkie.functions.base import BaseToolkit
from milkie.functions.openai_function import OpenAIFunction

class SampleToolKit(BaseToolkit):

    def testToolAdd(self, a: int, b: int) -> int:
        """将两个数字相加。

        Args:
            a: 要相加的第一个数字。
            b: 要相加的第二个数字。

        Returns:
            两个数字的和。
        """
        return a + b
    
    def testToolSub(self, a: int, b: int) -> int:
        r"""将两个数字相减。
        
        Args:
            a (int): 被减数。
            b (int): 减数。

        Returns:
            int: 两个数字的差。
        """
        return a - b

    def testToolFactorialDifference(self, a: int, b: int) -> int:
        r"""计算两个数字的阶乘差的绝对值。

        此函数首先计算两个给定整数的阶乘，然后计算它们的差的绝对值。
        这对于比较两个数字的阶乘大小很有用。

        Args:
            a (int): 第一个整数。
            b (int): 第二个整数。

        Returns:
            int: 两个数字的阶乘之差的绝对值。

        Examples:
            >>> testToolFactorialDifference(4, 3)
            18  # |4! - 3!| = |24 - 6| = 18
        """
        return abs(SampleToolKit.factorial(a) - SampleToolKit.factorial(b))

    def searchWebFromDuckDuckGo(
        self, query: str, max_results: int = 10
    ) -> str:
        r"""在互联网上搜索信息。

        此函数使用DuckDuckGo搜索引擎在互联网上搜索信息。

        Args:
            query (str): 要搜索的查询字符串
            max_results (int): 最大搜索结果数，默认为10

        Returns: 搜索结果
        """
    
        from duckduckgo_search import DDGS
        from requests.exceptions import RequestException

        ddgs = DDGS()
        responses: List[Dict[str, Any]] = []

        try:
            results = ddgs.text(keywords=query, max_results=max_results)
        except RequestException as e:
            responses.append({"error": f"duckduckgo search failed.{e}"})

        for i, result in enumerate(results, start=1):
            response = {
                "result_id": i,
                "title": result["title"],
                "description": result["body"],
                "url": result["href"],
            }
            responses.append(response)
        return json.dumps(responses, ensure_ascii=False)
        
    def getWebContentFromUrls(self, input_text: str) -> str:
        r"""从给定的文本中提取URL并获取网页内容。

        Args:
            input_text (str): 包含URL的文本。

        Returns: 网页内容
        """
        url_pattern = r'https?://[^\s<>"\'\]]+'
        urls = re.findall(url_pattern, input_text)

        if not urls:
            return "No valid URL found in the input text."

        clean_texts = []

        for url in urls:
            try:
                # 访问URL，获取HTML页面内容
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                html_content = response.text

                # 从HTML中提取出文章正文信息
                article = Article(url)
                article.set_html(html_content)
                article.parse()

                # 获取清理后的文章正文信息
                clean_text = article.text.strip()
                clean_texts.append(clean_text if clean_text else "No article content found.")

            except requests.RequestException as e:
                clean_texts.append(f"Error fetching the URL: {str(e)}")
            except Exception as e:
                clean_texts.append(f"Error processing the article: {str(e)}")

        return "\n\n".join(clean_texts)

    def factorial(n):
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.testToolAdd),
            OpenAIFunction(self.testToolSub),
            OpenAIFunction(self.testToolFactorialDifference),
            OpenAIFunction(self.searchWebFromDuckDuckGo),
            OpenAIFunction(self.getWebContentFromUrls),
        ]

    def getToolsDesc(self) -> str:
        tools = self.getTools()
        toolDescriptions = [tool.get_function_name() + ": " + tool.get_function_description() for tool in tools]
        return "\n".join(toolDescriptions)

if __name__ == "__main__":
    print(SampleToolKit().searchWebFromDuckDuckGo("拜仁"))