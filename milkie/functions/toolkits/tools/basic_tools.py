from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import os
import random
import re
import logging
import smtplib
import time
from urllib.parse import urlparse
import PyPDF2
from newspaper import Article
import markdown2
from typing import Any, Dict, List

import requests
from milkie.cache.cache_kv import GlobalCacheKVCenter
from milkie.config.config_robots_whitelist import getRobotPolicy, loadRobotPolicies
from milkie.functions.toolkits.tools.tool import Tool
from milkie.log import DEBUG, ERROR, INFO, WARNING
from milkie.utils.data_utils import preprocessHtml

logger = logging.getLogger(__name__)

class ToolSearchWebFromDuckDuckGo(Tool):
    
    def __init__(self):
        self.cacheMgr = GlobalCacheKVCenter.getCacheMgr("data/cache/", category='web_search', expireTimeByDay=1)

    def execute(
        self,
        query: str, 
        maxResults: int = 10
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

        ddgs = DDGS()
        responses: List[Dict[str, Any]] = []

        try:
            results = ddgs.text(keywords=query, max_results=maxResults)
        except Exception as e:
            responses.append({"error": f"duckduckgo search failed.{e}"})
            return None

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

class ToolSearchWebFromSougouWechat(Tool):

    def __init__(self):
        self.cacheMgr = GlobalCacheKVCenter.getCacheMgr("data/cache/", category='web_search', expireTimeByDay=1)

    def execute(self, query: str, maxResults: int = 10) -> str:
        pass

class ToolGetWebContentFromUrls(Tool):

    def __init__(self):
        self.lastAccessTime = {}
        self.cacheMgr = GlobalCacheKVCenter.getCacheMgr("data/cache/", category='web_content', expireTimeByDay=1)
        self.robotPolicies = loadRobotPolicies("config/robots.yaml")

    def execute(
        self,
        inputText: str
    ) -> str:
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
                # 设置下载状态为成功
                from newspaper.article import ArticleDownloadState
                article.download_state = ArticleDownloadState.SUCCESS
                article.parse()

                # 获取清理后的文章正文信息
                cleanText = article.text.strip()
                if not cleanText:
                    # 如果newspaper解析失败，使用简单的HTML清理
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(htmlContent, 'html.parser')
                    # 移除script和style标签
                    for script in soup(["script", "style"]):
                        script.decompose()
                    cleanText = soup.get_text(separator='\n').strip()
                
                cleanTexts.append(cleanText if cleanText else "No article content found.")

            except Exception as e:
                errorMsg = f"Error processing the article from {url}: {str(e)}"
                WARNING(logger, errorMsg)
                cleanTexts.append(errorMsg)

        return "\n\n".join(cleanTexts)

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
                    waitTime = min(5, robotPolicy.delay - timeSinceLastAccess)  
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

class ToolSendEmail(Tool):

    def __init__(self):
        pass

    def execute(
        self,
        to_email: str,
        subject: str,
        body: str
    ) -> str:
        """
        发送电子邮件给指定邮箱

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
            emailConfig = self.globalContext.globalConfig.getEmailConfig()
            
            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = emailConfig.username
            msg['To'] = to_email
            msg['Subject'] = subject

            # 处理不同类型的内容
            html_content = markdown2.markdown(body)
            msg.attach(MIMEText(html_content, 'html'))

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

class ToolGetHtmlContent(Tool):
    
    def execute(
        self,
        url: str
    ) -> str:
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

class ToolReadPdfContent(Tool):
    
    def execute(
        self,
        file_path: str
    ) -> str:
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

class ToolReadTxtContent(Tool):
    
    def execute(
        self,
        file_path: str
    ) -> str:
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

class ToolDownloadFileFromUrl(Tool):
    
    def execute(
        self,
        url: str,
        localDirectory: str = "data/pdf/"
    ) -> str:
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

if __name__ == "__main__":
    tool = ToolGetWebContentFromUrls()
    print(tool.execute("https://gzh.sogou.com/link?url=dn9a_-gY295K0Rci_xozVXfdMkSQTLW6cwJThYulHEtVjXrGTiVgS3hZnhri_N6Bg_6QFbXnNiHFlp-jtSY3U1qXa8Fplpd9aILA1MZiPLyr7wwt_HSME45tJvneg6RTWPcHmUejWGn7Y8mApH927kwS9HarzbSEVZX3ahogZu0Km75Ex2q45T5DlwTSbN8sBBSpqoZUXcS1xcw79erA7Z-QdkaDnp3mS9YdHKhvJ9nYfBa7oFPjcWbMMgYtVHjjNQwnEtRUz2ajotuuiAmOIg..&type=2&query=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&token=C6EC131AB41321CE9096BE9092A2BA10912A00ED67AD8E67"))