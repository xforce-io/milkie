import os
import requests
import PyPDF2
from datetime import datetime, timedelta
import json
# 路径常量
INTERESTED_FILE = "examples/paper_report/interested.json"
CRAWLERS_DATA_DIR = "dev/github/crawlers/data/papers/"
PAPER_REPORT_DIR = "dev/github/milkie/data/paper_report/"
RAW_DIR = "raw"
IMGS_DIR = "imgs"

# 关键词列表（不区分大小写）
interested_keywords = json.load(open(INTERESTED_FILE))
KEYWORDS = set(
        [keyword for keywords in interested_keywords.values() for keyword in keywords] +
        list(interested_keywords.keys())
)

def get_paper_filepaths(date, days, root):
    # 构建绝对路径
    DIR_WEBPAGES = os.path.join(root, CRAWLERS_DATA_DIR)
    DIR_RAWFILES = os.path.join(root, PAPER_REPORT_DIR)
    
    filepaths = []
    target_date = datetime.strptime(date, '%Y-%m-%d')
    target_date_start = target_date - timedelta(days=days)

    # 首先获取所有domain目录
    for domain in os.listdir(DIR_WEBPAGES):
        domain_path = os.path.join(DIR_WEBPAGES, domain)
        for time_dir in os.listdir(domain_path):
            try:
                dir_date = datetime.strptime(time_dir, '%Y-%m-%d')
                if dir_date >= target_date_start:
                    time_path = os.path.join(domain_path, time_dir)
                    if os.path.exists(time_path):
                        for root, dirs, files in os.walk(time_path):
                            for file in files:
                                if file.endswith('.txt'):
                                    file_path = os.path.join(root, file)
                                    filepaths.append(file_path)
            except ValueError:
                continue
    
    return filepaths, DIR_RAWFILES

def get_pdf_url(url):
    """将arxiv网页URL转换为PDF下载链接"""
    if 'arxiv.org' in url:
        # 移除 'abs' 替换为 'pdf'
        pdf_url = url.replace('abs', 'pdf')
        if not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url + '.pdf'
        return pdf_url
    return url

def contains_keywords(text: str) -> bool:
    """检查文本是否包含关键词（不区分大小写）"""
    text = text.lower()
    return any(keyword.lower() in text for keyword in KEYWORDS)

def process_papers(date, days, root):
    # 获取文件路径和输出目录
    filepaths, raw_dir_base = get_paper_filepaths(date, days, root)
    
    # 确保输出目录存在
    raw_dir = os.path.join(raw_dir_base, RAW_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析文件内容
            title = ''
            url = ''
            date_match = ''
            summary = ''
            arxiv_id = os.path.splitext(os.path.basename(filepath))[0]  # 从文件名获取arxiv_id
            
            # 提取标题
            title_parts = content.split('标题：')
            if len(title_parts) > 1:
                title = title_parts[1].split('\n')[0].strip()
            
            # 提取摘要
            summary_parts = content.split('摘要：')
            if len(summary_parts) > 1:
                summary = summary_parts[1].split('\n')[0].strip()
            
            # 提取发布日期
            date_parts = content.split('发布日期：')
            if len(date_parts) > 1:
                date_match = date_parts[1].split('\n')[0].strip()
                
            # 提取下载地址
            url_parts = content.split('下载地址：')
            if len(url_parts) > 1:
                url = url_parts[1].split('\n')[0].strip()
                
            if not url or not title or not date_match:
                continue

            # 检查标题和摘要是否包含关键词
            if not (contains_keywords(title) or contains_keywords(summary)):
                continue
                
            # 创建日期目录
            date_dir = os.path.join(raw_dir, date_match)
            os.makedirs(date_dir, exist_ok=True)
                
            # 检查PDF路径和元信息路径
            pdf_path = os.path.join(date_dir, f'{arxiv_id}.pdf')  # 使用arxiv_id作为文件名
            txt_path = os.path.join(date_dir, f'{arxiv_id}.txt')
            
            # PDF不存在时下载
            if not os.path.exists(pdf_path):
                # 获取PDF下载链接
                pdf_url = get_pdf_url(url)
                
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(pdf_url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    with open(pdf_path, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                        
                    # 检查文件大小
                    if os.path.getsize(pdf_path) < 10000:
                        raise RuntimeError(f"下载的PDF文件可能不完整: {pdf_url}")
                        
                except Exception as e:
                    print(f"下载PDF文件 {arxiv_id} 时出错: {str(e)}")
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    continue
            
            # 如果PDF存在但文本不存在，提取文本
            if os.path.exists(pdf_path) and not os.path.exists(txt_path):
                try:
                    # 提取文本
                    with open(pdf_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text = title + '\n' + url + '\n'
                        for page in reader.pages:
                            text += page.extract_text()
                            
                    # 保存文本到对应日期目录
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)
                except Exception as e:
                    print(f"提取文本时出错 {arxiv_id}: {str(e)}")

            print(f"成功处理文件: {arxiv_id}/{title}")
                    
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {str(e)}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python download.py <date> <days> <root>")
        print("Example: python download.py 2024-11-20 3 /path/to/root")
        sys.exit(1)
    
    date = sys.argv[1]
    days = int(sys.argv[2])
    root = sys.argv[3]
    process_papers(date, days, root)