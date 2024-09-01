import yaml

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
    
    # 压缩HTML
    html = str(soup)
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)
    
    return html.strip()