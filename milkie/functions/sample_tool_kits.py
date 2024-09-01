import json
import re
import requests
from newspaper import Article
from typing import Any, Dict, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time
import PyPDF2
import io

from milkie.context import Context, GlobalContext  # 新增导入
from milkie.functions.base import BaseToolkit
from milkie.functions.openai_function import OpenAIFunction
from milkie.cache.cache_kv import CacheKVMgr

class SampleToolKit(BaseToolkit):
    def __init__(self, globalContext: GlobalContext):  # 修改这里
        super().__init__()

        self.cacheMgr = CacheKVMgr("data/cache/")
        self.globalContext = globalContext  # 修改这里

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
        self, query: str, maxResults: int = 10
    ) -> str:
        r"""在互联��搜索信息。

        此函数使用DuckDuckGo搜索引擎在互联网上搜索信息。

        Args:
            query (str): 要搜索的查询字符串
            maxResults (int): 最大搜索结果数，默认为10

        Returns: 搜索结果
        """
        modelName = "duckduckgo_search"
        cacheKey = [{"query": query, "maxResults": maxResults}]
        
        cachedResult = self.cacheMgr.getValue(modelName, cacheKey)
        if cachedResult:
            return json.dumps(cachedResult, ensure_ascii=False)

        from duckduckgo_search import DDGS
        from requests.exceptions import RequestException

        ddgs = DDGS()
        responses: List[Dict[str, Any]] = []

        try:
            results = ddgs.text(keywords=query, max_results=maxResults)
        except RequestException as e:
            responses.append({"error": f"duckduckgo search failed.{e}"})

        for i, result in enumerate(results, start=1):
            response = {
                "resultId": i,
                "title": result["title"],
                "description": result["body"],
                "url": result["href"],
            }
            responses.append(response)
        
        self.cacheMgr.setValue(modelName, cacheKey, responses)
        return json.dumps(responses, ensure_ascii=False)

    def getWebContentFromUrls(self, inputText: str) -> str:
        r"""从给定的文本中提取URL并获取网页内容。

        Args:
            inputText (str): 包含URL的文本。

        Returns: 网页内容
        """
        urlPattern = r'https?://[^\s<>"\'\]]+'
        urls = re.findall(urlPattern, inputText)

        if not urls:
            return "No valid URL found in the input text."

        cleanTexts = []

        for url in urls:
            try:
                # 访问URL，获取HTML页面内容
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                htmlContent = response.text

                # 从HTML中提取出文章正文信息
                article = Article(url)
                article.set_html(htmlContent)
                article.parse()

                # 获取清理后的文章正文信息
                cleanText = article.text.strip()
                cleanTexts.append(cleanText if cleanText else "No article content found.")

            except requests.RequestException as e:
                cleanTexts.append(f"Error fetching the URL: {str(e)}")
            except Exception as e:
                cleanTexts.append(f"Error processing the article: {str(e)}")

        return "\n\n".join(cleanTexts)

    def sendEmail(self, to_email: str, subject: str, body: str) -> str:
        """
        发送电子邮件

        Args:
            to_email (str): 收件人邮箱地址
            subject (str): 邮件主题
            body (str): 邮件正文

        Returns:
            str: 发送结果描述

        Raises:
            Exception: 如果发送邮件过程中出现错误
        """
        try:
            # 获取邮件配置
            emailConfig = self.globalContext.globalConfig.getEmailConfig()  # 修改这里
            
            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = emailConfig.username
            msg['To'] = to_email
            msg['Subject'] = subject

            # 添加邮件正文
            msg.attach(MIMEText(body, 'plain'))

            # 连接到SMTP服务器并发送邮件
            with smtplib.SMTP(emailConfig.smtp_server, emailConfig.smtp_port) as server:
                server.starttls()
                server.login(emailConfig.username, emailConfig.password)
                server.send_message(msg)

            return f"success to send email to [{to_email}]"
        except Exception as e:
            return f"failed to send email[{str(e)}]"


    def runCodeInterpreter(self, instruction: str) -> str:
        r"""根据指令生成代码，并且用代码解释器执行代码。

        Args:
            instruction (str): 要执行的指令。

        Returns: 执行结果
        """
        from milkie.functions.code_interpreter import CodeInterpreter
        codeInterpreter = CodeInterpreter(self.globalContext)
        return codeInterpreter.execute(instruction)

    def factorial(n):
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def getHtmlContent(self, url: str) -> str:
        """
        获取指定URL的HTML页面内容,包含基本的反爬虫逻辑。

        Args:
            url (str): 要获取内容的网页URL

        Returns:
            str: HTML页面内容或错误信息

        Raises:
            Exception: 如果获取过程中出现错误
        """
        try:
            # 使用随机User-Agent
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
            ]
            headers = {'User-Agent': random.choice(user_agents)}

            # 添加随机延迟
            time.sleep(random.uniform(1, 3))

            # 发送请求
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return response.text
        except requests.RequestException as e:
            return f"Error fetching the URL: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def readPdfContent(self, file_path: str) -> str:
        """
        读取PDF文件的内容。

        Args:
            file_path (str): PDF文件的路径

        Returns:
            str: PDF文件的文本内容
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
                return content
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.testToolFactorialDifference),
            OpenAIFunction(self.searchWebFromDuckDuckGo),
            OpenAIFunction(self.getWebContentFromUrls),
            OpenAIFunction(self.sendEmail),
            OpenAIFunction(self.getHtmlContent),
            OpenAIFunction(self.readPdfContent),  # 添加新的PDF读取函数
        ]

    def getToolsDesc(self) -> str:
        tools = self.getTools()
        toolDescriptions = [tool.get_function_name() + ": " + tool.get_function_description() for tool in tools]
        return "\n".join(toolDescriptions)

if __name__ == "__main__":
    context = Context.createContext("config/global.yaml")

    #print(SampleToolKit().searchWebFromDuckDuckGo("拜仁"))
    print(SampleToolKit(context.globalContext).sendEmail("Freeman.xu@aishu.cn", "testHead", "testBody"))