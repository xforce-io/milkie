import os
from collections import defaultdict
from typing import Dict, List, Tuple

def get_topic2paper_path(date: str, root: str) -> str:
    """获取指定日期的 topic2paper.txt 文件路径"""
    return os.path.join(root, "dev/github/milkie/data/paper_report", date, "topic2paper.txt")

def read_topic_mappings(filepath: str) -> Dict[str, List[str]]:
    """读取并解析 topic2paper.txt 文件,返回 topic -> digest文件列表 的映射"""
    topic_to_digests = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(">>")
            if len(parts) != 2:
                continue
                
            topic, digest_path = parts[0].strip(), parts[1].strip()
            if topic == "None":
                continue
                
            # 将一个文件路径可能对应多个topic的情况处理好
            topics = [t.strip() for t in topic.split("/")]
            for t in topics:
                topic_to_digests[t].append(digest_path)
    
    return dict(topic_to_digests)

def read_digest_content(filepath: str) -> str:
    """读取 digest 文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading digest file {filepath}: {e}")
        return ""

def generate_preview(topic_mappings: Dict[str, List[str]]) -> str:
    """生成预览内容"""
    preview_content = []
    
    for topic, digest_paths in topic_mappings.items():
        preview_content.append(f"【{topic}】")
        for path in digest_paths:
            content = read_digest_content(path)
            if content:
                preview_content.append(content + "\n")
        preview_content.append("\n")  # 添加空行分隔不同主题
    
    return "\n".join(preview_content)

def write_preview(output_dir: str, content: str):
    """将预览内容写入文件"""
    preview_path = os.path.join(output_dir, "preview.txt")
    try:
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Preview file generated: {preview_path}")
    except Exception as e:
        print(f"Error writing preview file: {e}")

def process_topics(date: str, root: str):
    """主处理函数"""
    # 获取 topic2paper.txt 文件路径
    topic2paper_path = get_topic2paper_path(date, root)
    if not os.path.exists(topic2paper_path):
        print(f"Error: topic2paper.txt not found at {topic2paper_path}")
        return
    
    # 读取并解析映射关系
    topic_mappings = read_topic_mappings(topic2paper_path)
    if not topic_mappings:
        print("No valid topic mappings found")
        return
    
    # 生成预览内容
    preview_content = generate_preview(topic_mappings)
    
    # 写入预览文件
    output_dir = os.path.dirname(topic2paper_path)
    write_preview(output_dir, preview_content)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python topic_inverted.py <date> <root>")
        print("Example: python topic_inverted.py 2024-11-20 /path/to/root")
        sys.exit(1)
    
    date = sys.argv[1]
    root = sys.argv[2]
    process_topics(date, root)
