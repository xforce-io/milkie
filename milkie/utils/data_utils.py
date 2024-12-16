import yaml
import logging

logger = logging.getLogger(__name__)

def loadFromYaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

import re
from bs4 import BeautifulSoup, Comment

def preprocessHtml(htmlContent):
    soup = BeautifulSoup(htmlContent, 'html.parser')
    
    # 移除注释
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 移除所有<script>和<style>标签
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    
    # 保留主要内容区域
    main_content = soup.find('main')
    if main_content:
        soup = BeautifulSoup(str(main_content), 'html.parser')
    
    # 保留文章、作者、元数据和链接
    for tag in soup.find_all(['h3', 'a', 'time']):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['href', 'datetime']}
    
    # 移除多余的div，只保留必要的结构
    for div in soup.find_all('div'):
        if not div.find(['h3', 'a', 'time', 'p']):
            div.unwrap()
    
    # 移除所有类名和多余属性
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['href', 'datetime']}
    
    # 删除只包含其他标签而没有文本内容的标签
    def remove_empty_tags(soup):
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0 and len(tag.contents) > 0:
                tag.unwrap()
    
    # 多次应用删除操作，以处理嵌套的空标签
    for _ in range(3):  # 通常3次迭代足够处理大多数情况
        remove_empty_tags(soup)
    
    # 压缩HTML
    html = str(soup)
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)
    
    return html.strip()

def isBlock(blockType: str, blockContent: str) -> bool:
    if not blockContent or not blockType:
        return False
    normalized_content = blockContent.strip()
    return normalized_content.startswith(f"```{blockType}") or normalized_content.startswith(blockType)

def extractFromBlock(blockType: str, blockContent: str) -> str | None:
    content = re.sub(f"^```{blockType}|^{blockType}", '', blockContent)
    content = re.sub(r'```', '', content)
    return content.strip()

def extractBlock(blockType: str, blockContent: str) -> str | None:
    pattern = r'```%s\s*(.*?)\s*```' % (blockType if blockType else "")
    matches = re.findall(pattern, blockContent, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    return None

def escape(prompt :str) :
    # to prevent 'format' exception in get_template_vars
    return prompt.replace("{", "{{").replace("}", "}}")

def unescape(prompt :str) :
    return re.sub(r'\{{2,}', '{', re.sub(r'\}{2,}', '}', prompt))

def restoreVariablesInDict(data :dict, allDict :dict) -> dict:
    newDict = {}
    for argKey, argValue in data.items():
        key = restoreVariablesInStr(argKey, allDict)
        if isinstance(argValue, str):
            newDict[key] = restoreVariablesInStr(argValue, allDict)
    return newDict

def restoreVariablesInStr(data :str, allDict :dict):
    def recursive_lookup(data, key):
        keys = key.split('.')
        val = data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, list):
                try:
                    val = val[int(k)]
                except (ValueError, IndexError):
                    logger.error(f"Error looking up key: {key}")
                    return None
            else:
                logger.error(f"Error looking up key: {key}")
                return None
        return val

    from string import Formatter
    class NestedFormatter(Formatter):
        def get_field(self, fieldName, args, kwargs):
            return recursive_lookup(allDict, fieldName), fieldName

    return NestedFormatter().format(data)

def postRestoreVariablesInStr(data :str, allDict :dict) -> str:
    pattern = r'({[\w\.]+})'
    matches = re.findall(pattern, data, re.DOTALL)
    for match in matches:
        slot = restoreVariablesInStr(match, allDict).replace("\n", "//")
        data = data.replace(match, slot)
    return data

def wrapVariablesInStr(data: str) -> str:
    replacements = []
    pattern = r'({[\w\.]+})'
    for match in re.finditer(pattern, data):
        replacements.append((match.start(), match.end(), match.group()))
    for start, end, match_text in reversed(replacements):
        data = data[:start] + f"'{match_text}'" + data[end:]
    return data