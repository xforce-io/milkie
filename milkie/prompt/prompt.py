import glob
import os

PromptDir = "prompts/"
PromptPrefix = "prompt_"

class Prompt:
    def __init__(self, content :str):
        self.content = content

    def getContent(self) -> str:
        return self.content

class Loader:
    def load(name :str) -> str:
        pathPrompt = f"{PromptDir}{name}.txt"
        with open(pathPrompt, 'r') as file:
            content = file.read()
        return content

    def loadByPrefix(
            prefix :str=PromptPrefix, 
            directory :str=PromptDir) -> dict:
        filepaths = glob.glob(os.path.join(directory, f"{prefix}*"))
        filePostfix = ".txt"
        results = {}
        for filepath in filepaths:
            file_name = os.path.basename(filepath)
            with open(filepath, 'r') as file:
                file_name = file_name[len(prefix):]
                if file_name.endswith(filePostfix):
                    file_name = file_name[:-len(filePostfix)]
                results[file_name] = file.read()
        return results

GLoader = Loader()