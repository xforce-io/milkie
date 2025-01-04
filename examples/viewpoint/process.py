import os
import sys
from typing import Dict, List, Tuple

def readEventMap(eventFilePath: str) -> Dict[str, List[str]]:
    """读取事件到文件路径的映射"""
    eventMap = {}
    with open(eventFilePath, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split(">>")
            event, filepath = items[0].strip(), items[1].strip()
            if event == "None":
                continue

            if event not in eventMap:
                eventMap[event] = []
            eventMap[event].append(filepath)
    return eventMap

def extractViewpoint(filepath: str) -> str:
    """从文件中提取观点"""
    # 获取不带路径和扩展名的文件名
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "观点" in line:
                viewpoint = line.strip().replace("核心观点：", "").replace("观点：", "").strip()
                return f"{viewpoint} -> {filename}"
    return ""

def processViewpoints(eventMap: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """处理每个事件的观点"""
    # 按文件数量排序事件
    return sorted(
        [(event, [extractViewpoint(fp) for fp in filepaths]) 
         for event, filepaths in eventMap.items()],
        key=lambda x: len(x[1]),
        reverse=True
    )

def writeResults(resultPath: str, sortedViewpoints: List[Tuple[str, List[str]]]):
    """写入结果到文件"""
    with open(resultPath, "w", encoding="utf-8") as f:
        for event, viewpoints in sortedViewpoints:
            f.write(f"{event}\n---\n")
            for viewpoint in viewpoints:
                if viewpoint:  # 只写入非空观点
                    f.write(f"{viewpoint}\n")
            f.write("===\n\n")

def main(date: str):
    """主函数"""
    # 构建文件路径
    baseDir = os.path.join("data", "viewpoint", date)
    eventFilePath = os.path.join(baseDir, "event2filepath.txt")
    resultPath = os.path.join(baseDir, "viewpoint_result.txt")

    # 确保输入文件存在
    if not os.path.exists(eventFilePath):
        print(f"Error: Input file not found: {eventFilePath}")
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(resultPath), exist_ok=True)

    # 处理流程
    eventMap = readEventMap(eventFilePath)
    sortedViewpoints = processViewpoints(eventMap)
    writeResults(resultPath, sortedViewpoints)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process.py <date>")
        print("Example: python process.py 2024-11-20")
        sys.exit(1)

    main(sys.argv[1])
