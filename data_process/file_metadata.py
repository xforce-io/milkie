import os
import json
import datetime
from pathlib import Path

from milkie.agent.team.mrqa import MapReduceQA

SupportedFormat = ["pdf"]

def get_file_metadata(file_path, summaryAgent):
    file_stat = os.stat(file_path)

    metadata = {
        "name": os.path.basename(file_path),
        "size": file_stat.st_size,
    }

    response = summaryAgent.execute(
        input="@filepath:%s" % file_path, 
        task="写一段概要总结，不超过 200 字")
    if response is not None:
        metadata["summary"] = response.response

    #response = chat(
    #    llm=self.roleAgent.context.globalContext.settings.llm, 
    #    systemPrompt=self.roleAgent.systemPrompt,
    #    prompt=prompt, 
    #    promptArgs={})

    for format in SupportedFormat:
        if file_path.lower().endswith(format):
            return metadata
    return None

def get_all_files_metadata(directory):
    summaryAgent = MapReduceQA()
    for root, _, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            yield get_file_metadata(file_path, summaryAgent)

def main():
    directory = "/Users/xupeng/Documents/eb/materials"
    output_file = "/tmp/paper.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for metadata in get_all_files_metadata(directory):
            if metadata == None :
                continue
            
            json.dump(metadata, f, ensure_ascii=False)
            f.write('\n')

    print(f"文件元数据已保存到 {output_file}")

if __name__ == "__main__":
    main()
