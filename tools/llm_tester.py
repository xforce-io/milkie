import os, glob
from transformers import AutoModelForCausalLM, AutoTokenizer

Models = [
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-14B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-13B-Chat/",
]

def getPrompts():
    directory_path = 'prompts'
    file_prefix = 'prompt_'
    file_postfix = '.txt'

    file_paths = glob.glob(os.path.join(directory_path, file_prefix + '*'))

    results = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r') as file:
            file_name = file_name[len(file_prefix):]
            if file_name.endswith(file_postfix):
                file_name = file_name[:-len(file_postfix)]

            item = {}
            item["name"] = file_name
            item["prompt"] = file.read()
            results += [item]
    return results

Prompts = getPrompts()

def testModel(modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelName, device_map="auto", trust_remote_code=True).eval()
    for promptItem in Prompts:
        print("=================================================")
        print("model:", modelName)
        print("prompt:", promptItem["name"])
        print("-------------------------------------------------")
        response = model.chat(tokenizer, promptItem["prompt"])
        print(response)

if __name__ == "__main__":
    for model in Models:
        testModel(model)