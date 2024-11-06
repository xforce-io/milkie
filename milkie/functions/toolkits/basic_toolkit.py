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
import markdown2

from milkie.context import Context
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.openai_function import OpenAIFunction
from milkie.cache.cache_kv import CacheKVMgr
from milkie.global_context import GlobalContext
from milkie.utils.data_utils import preprocessHtml
from milkie.log import ERROR, INFO, DEBUG, WARNING
from milkie.functions.code_interpreter import CodeInterpreter

logger = logging.getLogger(__name__)

class BasicToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

        self.cacheMgr = CacheKVMgr("data/cache/", expireTimeByDay=1)
        self.robotPolicies = loadRobotPolicies("config/robots.yaml")
        self.lastAccessTime = {} 

        self.codeInterpreter = CodeInterpreter(self.globalContext)

    def searchWebFromDuckDuckGo(
        self, query: str, maxResults: int = 10
    ) -> str:
        r"""在互联网搜索信息

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
        r"""从给定的文本中提取 urls， 并且获取这些 urls 对应的网页内容。

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
                errorMsg = f"Error processing the article from {url}: {str(e)}"
                ERROR(logger, errorMsg)
                cleanTexts.append(errorMsg)
                raise RuntimeError(errorMsg)

        return "\n\n".join(cleanTexts)

    def sendEmail(self, to_email: str, subject: str, body: str, content_type: str = "plain") -> str:
        """
        发送电子邮件给指定邮箱

        Args:
            to_email (str): 收件人邮箱地址
            subject (str): 邮件主题
            body (str): 邮件正文
            content_type (str): 内容类型，可选值为 "plain"、"html" 或 "markdown"，默认为 "plain"

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

            # 处理不同类型的内容
            if content_type == "html":
                msg.attach(MIMEText(body, 'html'))
            elif content_type == "markdown":
                html_content = markdown2.markdown(body)
                msg.attach(MIMEText(html_content, 'html'))
            else:  # 默认为纯文本
                msg.attach(MIMEText(body, 'plain'))

            # 连接到SMTP服务器并发送邮件
            with smtplib.SMTP(emailConfig.smtp_server, emailConfig.smtp_port) as server:
                server.starttls()
                server.login(emailConfig.username, emailConfig.password)
                server.send_message(msg)

            return f"成功发送邮件至 [{to_email}]"
        except Exception as e:
            errorMsg = f"Error sending email[{subject}]: {str(e)}"
            ERROR(logger, errorMsg)
            raise RuntimeError(errorMsg)

    def getHtmlContent(self, url: str) -> str:
        """
        获取指定URL的HTML内容。

        Args:
            url (str): 要获取内容的网页URL

        Returns:
            str: HTML页面内容

        Raises:
            RuntimeError: 如果在重试后仍然无法获取内容
        """
        for attempt in range(2):  # 最多尝试两次
            try:
                content = self._fetchUrl(url)
                return content
            except Exception as e:
                if attempt == 0:  # 第一次尝试失败
                    WARNING(logger, f"First attempt to get HTML content failed [{url}], retrying: {str(e)}")
                    time.sleep(2)  # 等待2秒后重试
                else:  # 第二次尝试也失败
                    errorMsg = f"Failed to get HTML content after retry [{url}]: {str(e)}"
                    ERROR(logger, errorMsg)
                    raise RuntimeError(errorMsg)

    def readPdfContent(self, file_path: str) -> str:
        """
        读取本地PDF文件的内容。

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
            errorMsg = f"Error reading PDF[{file_path}]: {str(e)}"
            ERROR(logger, errorMsg)
            raise RuntimeError(errorMsg)

    def readTxtContent(self, file_path: str) -> str:
        """
        读取本地TXT文件的内容。

        Args:
            file_path (str): TXT文件的路径

        Returns:
            str: TXT文件的文本内容
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            errorMsg = f"Error reading TXT[{file_path}]: {str(e)}"
            ERROR(logger, errorMsg)
            raise RuntimeError(errorMsg)

    def downloadFileFromUrl(self, url: str, localDirectory: str = "data/pdf/") -> str:
        """
        从指定URL下载文件并保存到本地目录。

        Args:
            url (str): 要下载的文件的URL
            localDirectory (str): 保存下载文件的本地目录，默认为 'data/pdf/'

        Returns:
            str: 下载文件的本地路径或错误信息
        """
        try:
            # 从URL中提取文件名
            parsed_url = urlparse(url)
            fileName = os.path.basename(parsed_url.path)
            if not fileName:
                fileName = "downloaded_file"
            
            # 确保本地目录存在
            os.makedirs(localDirectory, exist_ok=True)
            
            # 构建完整的本地文件路径
            localPath = os.path.join(localDirectory, fileName)
            
            # 检查文件是否已经存在
            if os.path.exists(localPath):
                INFO(logger, f"File already exists: {localPath}")
                return localPath
            
            # 使用 requests 直接下载文件
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 如果请求不成功则抛出异常
            
            # 保存文件
            with open(localPath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            # 验证文件是否下载成功
            if os.path.getsize(localPath) == 0:
                raise Exception("Downloaded file is empty")
            
            return localPath
        except requests.RequestException as e:
            error_msg = f"Error downloading file from {url}: {str(e)}"
            ERROR(logger, error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while downloading file from {url}: {str(e)}"
            ERROR(logger, error_msg)
            raise RuntimeError(error_msg)

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.searchWebFromDuckDuckGo),
            OpenAIFunction(self.getWebContentFromUrls),
            OpenAIFunction(self.sendEmail),
            OpenAIFunction(self.getHtmlContent),
            OpenAIFunction(self.readPdfContent),
            OpenAIFunction(self.readTxtContent),
            OpenAIFunction(self.downloadFileFromUrl),
        ]

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
                    DEBUG(logger, f"Waiting {waitTime:.2f} seconds before accessing {url}")
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
        except requests.Timeout as e:
            errorMsg = f"Request {url} timeout"
            ERROR(logger, errorMsg)
            raise RuntimeError(f"Error: {errorMsg}")
        except requests.ConnectionError as e:
            errorMsg = f"Connection {url} error"
            ERROR(logger, errorMsg)
            raise RuntimeError(f"Error: {errorMsg}")
        except requests.RequestException as e:
            errorMsg = f"Request {url} exception"
            ERROR(logger, errorMsg)
            raise RuntimeError(f"Error: {errorMsg}")
        except Exception as e:
            errorMsg = f"Unexpected error {url}"
            ERROR(logger, errorMsg)
            raise RuntimeError(f"Error: {errorMsg}")

if __name__ == "__main__":
    context = Context.create("config/global.yaml")

    #print(SampleToolKit().searchWebFromDuckDuckGo("拜仁"))
    print(BasicToolkit(context.getGlobalContext()).getToolsDesc())