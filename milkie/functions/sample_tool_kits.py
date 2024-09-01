import json
import re
from urllib.parse import urlparse
import requests
from newspaper import Article
from typing import Any, Dict, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time
import PyPDF2
import logging
import os
from milkie.config.config_robots_whitelist import loadRobotPolicies, getRobotPolicy

from milkie.context import Context, GlobalContext
from milkie.functions.base import BaseToolkit
from milkie.functions.openai_function import OpenAIFunction
from milkie.cache.cache_kv import CacheKVMgr
from milkie.utils.data_utils import preprocessHtml

logger = logging.getLogger(__name__)

class SampleToolKit(BaseToolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__()

        self.cacheMgr = CacheKVMgr("data/cache/", expireTimeByDay=1)
        self.globalContext = globalContext
        self.robotPolicies = loadRobotPolicies("config/robots.yaml")
        self.lastAccessTime = {} 

    def testToolFactorialDifference(self, a: int, b: int) -> int:
        r"""计算两个数字的阶乘差的绝对值。

        此函数首先计算两给定整数的阶乘，然后计算它们的差的绝对值。
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
        return abs(SampleToolKit._factorial(a) - SampleToolKit._factorial(b))

    def searchWebFromDuckDuckGo(
        self, query: str, maxResults: int = 10
    ) -> str:
        r"""在互联网搜索信息。

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
                htmlContent = self._fetchUrl(url)

                # 从HTML中提取出文章正文信息
                article = Article(url)
                article.set_html(htmlContent)
                article.parse()

                # 获取清理后的文章正文信息
                cleanText = article.text.strip()
                cleanTexts.append(cleanText if cleanText else "No article content found.")

            except Exception as e:
                error_msg = f"Error processing the article from {url}: {str(e)}"
                logger.error(error_msg)
                cleanTexts.append(error_msg)

        return "\n\n".join(cleanTexts)

    def sendEmail(self, to_email: str, subject: str, body: str) -> str:
        """
        发送电子邮件

        Args:
            to_email (str): 收件人邮箱地
            subject (str): 邮件主题
            body (str): 邮件正文

        Returns:
            str: 发送结果描述

        Raises:
            Exception: 如果发送邮件过程中出现错误
        """
        try:
            # 获取邮件配置
            emailConfig = self.globalContext.globalConfig.getEmailConfig()
            
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

    def getHtmlContent(self, url: str) -> str:
        """
        获取指定URL的HTML页面内容。

        Args:
            url (str): 要获取内容的网页URL

        Returns:
            str: HTML页面内容或错误信息
        """
        return self._fetchUrl(url)

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

    def downloadFileFromUrl(self, url: str, local_directory: str = "./") -> str:
        """
        从指定URL下载文件并保存到本地目录。

        Args:
            url (str): 要下载的文件的URL
            local_directory (str): 保存下载文件的本地目录，默认为 './'

        Returns:
            str: 下载文件的本地路径或错误信息
        """
        try:
            # 从URL中提取文件名
            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)
            if not file_name:
                file_name = "downloaded_file"
            
            # 确保本地目录存在
            os.makedirs(local_directory, exist_ok=True)
            
            # 构建完整的本地文件路径
            local_path = os.path.join(local_directory, file_name)
            
            # 使用 requests 直接下载文件
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 如果请求不成功则抛出异常
            
            # 保存文件
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            # 验证文件是否下载成功
            if os.path.getsize(local_path) == 0:
                raise Exception("Downloaded file is empty")
            
            return local_path
        except requests.RequestException as e:
            error_msg = f"Error downloading file from {url}: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error while downloading file from {url}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.testToolFactorialDifference),
            OpenAIFunction(self.searchWebFromDuckDuckGo),
            OpenAIFunction(self.getWebContentFromUrls),
            OpenAIFunction(self.sendEmail),
            OpenAIFunction(self.getHtmlContent),
            OpenAIFunction(self.readPdfContent),
            OpenAIFunction(self.downloadFileFromUrl),  # 添加新的下载文件函数
        ]

    def getToolsDesc(self) -> str:
        tools = self.getTools()
        toolDescriptions = [tool.get_function_name() + ": " + tool.get_function_description() for tool in tools]
        return "\n".join(toolDescriptions)

    def _factorial(n):
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def _fetchUrl(self, url: str, headers: Dict[str, str] = None, timeout: int = 10) -> str:
        """
        获取指定URL的内容，包含缓存、机器人策略控制和基本的反爬虫逻辑。

        Args:
            url (str): 要获取内容的网页URL
            headers (Dict[str, str], optional): 请求头
            timeout (int, optional): 请求超时时间，默认为10秒

        Returns:
            str: 网页内容或错误信息

        Raises:
            Exception: 如果获取过程中出现错误或访问不被允许
        """
        # 尝试从缓存获取内容
        cachedContent = self.cacheMgr.getValue("urlContent", [{"url": url}])
        if cachedContent:
            return preprocessHtml(cachedContent)

        try:
            # 检查访问策略
            robotPolicy = getRobotPolicy(url, self.robotPolicies)
            if not robotPolicy.allowed:
                raise Exception(f"Access to {url} is not allowed according to the robot policy.")
            
            # 检查是否需要等待
            currentTime = time.time()
            parsed_url = urlparse(url)
            urlKey = f"{parsed_url.netloc}{parsed_url.path}"
            if urlKey in self.lastAccessTime:
                timeSinceLastAccess = currentTime - self.lastAccessTime[urlKey]
                if timeSinceLastAccess < robotPolicy.delay:
                    waitTime = robotPolicy.delay - timeSinceLastAccess
                    logger.info(f"Waiting {waitTime:.2f} seconds before accessing {url}")
                    time.sleep(waitTime)  # 仍然需要等待，但只在必要时等待

            # 更新最后访问时间
            self.lastAccessTime[urlKey] = currentTime

            # 使用随机User-Agent（如果没有提供headers）
            if not headers:
                userAgents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
                ]
                headers = {'User-Agent': random.choice(userAgents)}

            # 发送请求
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            content = response.text
            
            # 将内容存入缓存
            self.cacheMgr.setValue("urlContent", [{"url": url}], content)

            return preprocessHtml(content)
        except requests.RequestException as e:
            errorMsg = f"Error fetching the URL: {str(e)}"
            logger.error(errorMsg)
            return errorMsg
        except Exception as e:
            errorMsg = f"Unexpected error: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

if __name__ == "__main__":
    context = Context.createContext("config/global.yaml")

    #print(SampleToolKit().searchWebFromDuckDuckGo("拜仁"))
    print(SampleToolKit(context.globalContext).sendEmail("Freeman.xu@aishu.cn", "testHead", "testBody"))