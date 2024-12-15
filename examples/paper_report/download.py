import os
import json
import requests
import PyPDF2
from datetime import datetime

DIR_WEBPAGES = "/Users/xupeng/lab/crawlers/data/paper/"
DIR_RAWFILES = "/Users/xupeng/dev/github/milkie/data/paper_report/"

def get_paper_filepaths(date):
    filepaths = []
    target_date = datetime.strptime(date, '%Y-%m-%d')
    
    # 遍历DIR_WEBPAGES下的时间目录
    for time_dir in os.listdir(DIR_WEBPAGES):
        try:
            dir_date = datetime.strptime(time_dir, '%Y-%m-%d')
            if dir_date >= target_date:
                time_path = os.path.join(DIR_WEBPAGES, time_dir)
                # 遍历domain目录
                for domain in os.listdir(time_path):
                    domain_path = os.path.join(time_path, domain)
                    for root, dirs, files in os.walk(domain_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            filepaths.append(file_path)
        except ValueError:
            continue
    return filepaths

def get_pdf_url(url):
    """将arxiv网页URL转换为PDF下载链接"""
    if 'arxiv.org' in url:
        # 移除 'abs' 替换为 'pdf'
        pdf_url = url.replace('abs', 'pdf')
        if not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url + '.pdf'
        return pdf_url
    return url

def process_papers(date):
    # 确保输出目录存在
    raw_dir = os.path.join(DIR_RAWFILES, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    filepaths = get_paper_filepaths(date)
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析文件内容
            title = ''
            url = ''
            date_match = ''
            
            # 提取标题
            title_match = content.split('标题：')
            if len(title_match) > 1:
                title = title_match[1].split('\n')[0].strip()
                if title.startswith('Paper page -'):
                    title = title.replace('Paper page -', '').strip()
            
            # 提取发布日期
            date_parts = content.split('发布日期：')
            if len(date_parts) > 1:
                date_match = date_parts[1].split('\n')[0].strip()
                
            # 提取下载地址
            url_match = content.split('下载地址：')
            if len(url_match) > 1:
                url = url_match[1].split('\n')[0].strip()
                
            if not url or not title or not date_match:
                continue
                
            # 创建日期目录
            date_dir = os.path.join(raw_dir, date_match)
            os.makedirs(date_dir, exist_ok=True)
                
            # 检查PDF是否已存在
            pdf_path = os.path.join(date_dir, f'{title}.pdf')
            if os.path.exists(pdf_path):
                continue
                
            # 获取PDF下载链接
            pdf_url = get_pdf_url(url)
            
            # 下载PDF
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
                    
                # 提取文本
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                        
                # 保存文本到对应日期目录
                txt_path = os.path.join(date_dir, f'{title}.txt')
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

                print(f"成功下载并保存文件: {title}")
                    
            except Exception as e:
                print(f"处理文件 {title} 时出错: {str(e)}")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {str(e)}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("请提供日期参数，格式为 YYYY-MM-DD")
        sys.exit(1)
    
    date = sys.argv[1]
    process_papers(date)
