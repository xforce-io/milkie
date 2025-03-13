import sys
import requests
from scholarly import scholarly, ProxyGenerator
import datetime
import json
from urllib.parse import quote
import time
from typing import List, Dict, Optional, Tuple
import feedparser
import urllib.parse
import pathlib
from PyPDF2 import PdfReader
import math
import os

# 路径相关的常量配置
CACHE_DIR = pathlib.Path('data/research_topic')
PDF_DIR = pathlib.Path('data/papers')
CITATIONS_CACHE_FILE = CACHE_DIR / 'citations.txt'
ARXIV_API_BASE_URL = 'http://export.arxiv.org/api/query'
CACHE_EXPIRY_DAYS = 7

# Scholarly 相关配置
MAX_RETRIES = 3
RETRY_DELAY = 5
SCHOLARLY_DELAY = 3

# 代理配置
PROXY_HOST = "brd.superproxy.io"
PROXY_PORT = "33335"
PROXY_USER = "brd-customer-hl_bbaae1f1-zone-residential_proxy1"
PROXY_PASS = "wbsqg5oex6jt"

def ensure_dir_exists(path: pathlib.Path):
    """确保目录存在，如果不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True)

def download_from_arxiv_id(arxiv_id: str) -> str:
    """
    根据 ArXiv ID 下载论文并转换为文本
    
    Args:
        arxiv_id (str): ArXiv 论文 ID
    
    Returns:
        str: 生成的文本文件的绝对路径
    """
    # 确保目录存在
    ensure_dir_exists(PDF_DIR)
    
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    txt_path = PDF_DIR / f"{arxiv_id}.txt"
    
    # 如果文本文件已存在，直接返回路径
    if txt_path.exists():
        print(f"Text file already exists for {arxiv_id}")
        return str(txt_path.absolute())
    
    # 如果 PDF 不存在，下载它
    if not pdf_path.exists():
        print(f"Downloading PDF for {arxiv_id}...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF for {arxiv_id}")
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF downloaded to {pdf_path}")
    else:
        print(f"PDF already exists for {arxiv_id}")
    
    # 将 PDF 转换为文本
    print(f"Converting PDF to text for {arxiv_id}...")
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        
        # 保存文本文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
        
        print(f"Text file created at {txt_path}")
        return str(txt_path.absolute())
        
    except Exception as e:
        print(f"Error converting PDF to text: {e}")
        if txt_path.exists():
            txt_path.unlink()  # 删除可能部分创建的文本文件
        raise

def get_related_papers(keywords: str) -> List[Dict]:
    """
    获取相关论文并下载其 PDF 和文本内容
    
    Args:
        keywords (str): 搜索关键词
    
    Returns:
        List[Dict]: 包含论文信息的字典列表
    """
    if not isinstance(keywords, str) or not keywords.strip():
        raise ValueError("Keywords must be a non-empty string")
        
    keywords = keywords.strip()
    print(f"Searching papers for keywords: {keywords}")
    papers = query_kw_from_arxiv(keywords)
    
    # 确保返回的是列表
    if not papers:
        return []
        
    for i, paper in enumerate(papers, 1):
        try:
            print(f"\nProcessing paper {i}/{len(papers)}: {paper['title'][:100]}...")
            paper['txt_path'] = download_from_arxiv_id(paper['arxiv_id'])
        except Exception as e:
            print(f"Error processing paper {paper['arxiv_id']}: {e}")
            continue
    
    return papers

def load_citations_cache() -> Dict[str, Dict]:
    """
    从缓存文件加载引用数据
    """
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)
    
    if not CITATIONS_CACHE_FILE.exists():
        return {}
    
    try:
        with open(CITATIONS_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading citations cache: {e}")
        return {}

def save_citations_cache(cache: Dict[str, Dict]):
    """
    保存引用数据到缓存文件
    """
    try:
        with open(CITATIONS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving citations cache: {e}")

def check_proxy(proxy_url: str) -> bool:
    """
    检查代理是否可用
    """
    try:
        response = requests.get('https://scholar.google.com', 
                              proxies={'http': proxy_url, 'https': proxy_url},
                              timeout=5)
        return response.status_code == 200
    except:
        return False

def setup_scholarly_proxy():
    """
    设置 scholarly 的代理
    使用 Bright Data 代理
    """
    try:
        # 构建代理URL
        proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        # 测试代理是否可用，禁用 SSL 验证
        response = requests.get('https://scholar.google.com', proxies=proxies, timeout=5, verify=False)
        if response.status_code == 200:
            print("Successfully set up Bright Data proxy")
            return proxies
    except Exception as e:
        print(f"Failed to set up Bright Data proxy: {e}")

    print("Warning: No working proxy found. Citations might not be available.")
    return None

def get_citation_count(title: str, retries: int = MAX_RETRIES) -> Tuple[int, bool]:
    """
    获取论文的引用次数
    
    Args:
        title (str): 论文标题
        retries (int): 重试次数
    
    Returns:
        Tuple[int, bool]: (引用次数, 是否成功)
    """
    for retry in range(retries):
        try:
            search_query = scholarly.search_pubs(title)
            pub = next(search_query, None)
            if pub:
                return pub['num_citations'], True
            return 0, True
        except Exception as e:
            if retry == retries - 1:  # 最后一次重试
                print(f"Failed to get citations after {retries} retries: {e}")
                return 0, False
            print(f"Retry {retry + 1}/{retries} after {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    return 0, False

def update_paper_citations(paper: Dict, citations_cache: Dict[str, Dict], 
                         cache_key: str) -> bool:
    """
    更新论文的引用信息并立即保存缓存
    
    Args:
        paper (Dict): 论文信息
        citations_cache (Dict): 引用缓存
        cache_key (str): 缓存键
    
    Returns:
        bool: 缓存是否更新
    """
    # 检查缓存
    if cache_key in citations_cache:
        cached_data = citations_cache[cache_key]
        cache_date = datetime.datetime.fromisoformat(cached_data['cache_date'])
        if (datetime.datetime.now() - cache_date).days <= CACHE_EXPIRY_DAYS:
            paper['citations'] = cached_data['citations']
            return False
    
    # 获取新的引用数据
    citations, success = get_citation_count(paper['title'])
    if success:
        paper['citations'] = citations
        citations_cache[cache_key] = {
            'citations': citations,
            'cache_date': datetime.datetime.now().isoformat(),
            'title': paper['title'],
            'arxiv_id': paper['arxiv_id']
        }
        # 立即保存缓存
        save_citations_cache(citations_cache)
        return True
    
    # 如果获取失败但有缓存，使用缓存的数据
    if cache_key in citations_cache:
        paper['citations'] = citations_cache[cache_key]['citations']
    else:
        paper['citations'] = 0
    
    return False

def get_citation_count_from_semantic_scholar(paper_title: str, paper_id: str = None) -> int:
    """
    使用Semantic Scholar API获取论文的引用次数
    
    Args:
        paper_title (str): 论文标题
        paper_id (str, optional): 论文的ID，如arxiv ID
    
    Returns:
        int: 引用次数
    """
    try:
        if paper_id and paper_id.startswith("arXiv:"):
            # 如果提供了arxiv ID，直接使用ID查询
            arxiv_id = paper_id.replace("arXiv:", "")
            api_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"
        else:
            # 否则使用标题搜索
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": paper_title,
                "fields": "citationCount",
                "limit": 1
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('data') and len(data['data']) > 0:
                return data['data'][0].get('citationCount', 0)
            return 0
            
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data.get('citationCount', 0)
    except Exception as e:
        print(f"Error fetching citation count from Semantic Scholar: {e}")
        return 0

def update_paper_citations_semantic_scholar(paper: Dict, citations_cache: Dict[str, Dict], 
                         cache_key: str) -> bool:
    """
    使用Semantic Scholar API更新论文的引用信息并立即保存缓存
    
    Args:
        paper (Dict): 论文信息
        citations_cache (Dict): 引用缓存
        cache_key (str): 缓存键
    
    Returns:
        bool: 缓存是否更新
    """
    # 检查缓存
    if cache_key in citations_cache:
        cached_data = citations_cache[cache_key]
        cache_date = datetime.datetime.fromisoformat(cached_data['cache_date'])
        if (datetime.datetime.now() - cache_date).days <= CACHE_EXPIRY_DAYS:
            paper['citations'] = cached_data['citations']
            return False
    
    # 获取新的引用数据
    paper_id = f"arXiv:{paper['arxiv_id']}" if 'arxiv_id' in paper else None
    citations = get_citation_count_from_semantic_scholar(paper['title'], paper_id)
    
    paper['citations'] = citations
    citations_cache[cache_key] = {
        'citations': citations,
        'cache_date': datetime.datetime.now().isoformat(),
        'title': paper['title'],
        'arxiv_id': paper.get('arxiv_id', '')
    }
    # 立即保存缓存
    save_citations_cache(citations_cache)
    return True

def query_kw_from_arxiv(query: str) -> List[Dict]:
    """
    从 ArXiv 搜索关键词并返回处理后的结果
    
    Args:
        query (str): 搜索关键词
    
    Returns:
        List[Dict]: 处理后的结果列表，按权重排序
    """
    # 设置代理
    proxy_available = setup_scholarly_proxy()
    
    # 加载引用缓存
    citations_cache = load_citations_cache()
    
    # 构建 ArXiv API URL
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': 100,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    search_url = f"{ARXIV_API_BASE_URL}?{urllib.parse.urlencode(params)}"
    print(f"Searching ArXiv with URL: {search_url}")
    
    # 发送请求获取搜索结果
    try:
        if proxy_available:
            response = requests.get(search_url, proxies=proxy_available)
        else:
            response = requests.get(search_url)
        feed = feedparser.parse(response.text)
    except Exception as e:
        print(f"Error accessing ArXiv: {e}, trying without proxy...")
        response = requests.get(search_url)
        feed = feedparser.parse(response.text)
    
    results = []
    total_entries = len(feed.entries)
    print(f"Found {total_entries} papers in search results")
    
    for i, entry in enumerate(feed.entries, 1):
        try:
            # 获取标题
            title = entry.title.replace('\n', ' ').strip()
            
            # 获取摘要
            abstract = entry.summary.replace('\n', ' ').strip()
            
            # 获取论文ID
            arxiv_id = entry.id.split('/abs/')[-1]
            
            # 获取日期
            date = datetime.datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ').date()
            
            paper_info = {
                "resultId": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "description": abstract,
                "url": entry.link,
                "date": date.isoformat()
            }
            results.append(paper_info)
            print(f"Successfully processed paper {i}/{total_entries}: {title[:100]}...")
        except Exception as e:
            print(f"Error processing paper: {str(e)}")
            continue
    
    print(f"\nFound {len(results)} papers, getting citations...")
    
    # 获取引用信息 - 使用Semantic Scholar API
    processed_papers = []
    for i, paper in enumerate(results, 1):
        try:
            cache_key = f"{paper['arxiv_id']}_{paper['title']}"
            # 使用Semantic Scholar替代scholarly
            update_paper_citations_semantic_scholar(paper, citations_cache, cache_key)
            print(f"Citations for paper {i}/{len(results)}: {paper['citations']} ({paper['title'][:50]}...)")
            
            # 计算权重并添加到处理后的列表中
            paper_date = datetime.date.fromisoformat(paper['date'])
            days_old = (datetime.date.today() - paper_date).days
            paper['weight'] = (paper['citations'] + 1) * (1 / math.log(days_old + 2))  # 加1避免全为0
            processed_papers.append(paper)
            
        except Exception as e:
            print(f"Error updating citations for paper {i}/{len(results)}: {str(e)}")
            # 处理错误情况
            paper_date = datetime.date.fromisoformat(paper['date'])
            days_old = (datetime.date.today() - paper_date).days
            paper['citations'] = 0
            paper['weight'] = 1 / math.log(days_old + 2)  # 使用基础权重
            processed_papers.append(paper)
            continue
    
    # 按权重排序
    processed_papers.sort(key=lambda x: x['weight'], reverse=True)
    
    print(f"\nReturning {len(processed_papers)} papers after sorting by weight")
    return processed_papers

def main(keywords: str = None) -> List[Dict]:
    """
    主函数，处理论文搜索和下载
    
    Args:
        keywords (str, optional): 搜索关键词，可以从环境变量或参数获取
    
    Returns:
        List[Dict]: 处理后的论文列表
    """
    # 如果参数为 None，尝试从环境变量获取
    if keywords is None:
        keywords = os.environ.get('keywords', '')
    
    # 确保关键词非空
    if not isinstance(keywords, str) or not keywords.strip():
        raise ValueError("Keywords must be a non-empty string")
    
    try:
        papers = get_related_papers(keywords)
        if not papers:
            print("No papers found")
        return papers
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    # 命令行方式运行
    if len(sys.argv) > 1:
        keywords = sys.argv[1]
        
    try:
        if not keywords:
            print("Usage: python process.py 'your search keywords'")
            print("Or set keywords environment variable")
            sys.exit(1)
            
        papers = main(keywords)
        print(json.dumps(papers, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)