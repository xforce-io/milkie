import os, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from enum import Enum

ModelNames = [
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-14B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-13B-Chat/",
]

class TypeModel(Enum):
    Qwen = 1
    Baichuan = 2

    @staticmethod
    def detectModelType(modelName :str):
        if modelName.find("qwen") >= 0:
            return TypeModel.Qwen
        elif modelName.find("baichuan") >= 0:
            return TypeModel.Baichuan
        else:
            return None

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
        modelType = TypeModel.detectModelType(modelName)
        
        print("=================================================")
        print("model type:", modelType)
        print("model:", modelName)
        print("prompt:", promptItem["name"])
        print("-------------------------------------------------")
        if modelType == TypeModel.Qwen:
            response, history = model.chat(
                tokenizer, 
                promptItem["prompt"], 
                history=None)
        elif modelType == TypeModel.Baichuan:
            model.generation_config = GenerationConfig.from_pretrained(modelName)
            messages = []
            messages.append({"role": "user", "content": "%s" % promptItem["prompt"]})
            response = model.chat(tokenizer, messages)
        else:
            raise Exception("Unknown model type")
        print(response)

if __name__ == "__main__":
    for model in ModelNames:
        testModel(model)