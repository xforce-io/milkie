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
    Sep = "~"
    
    def load(name :str) -> str:
        pathPrompt = f"{PromptDir}{name}.txt"
        with open(pathPrompt, 'r') as file:
            content = file.read()
        return content

    def loadByPrefix(
            directory :str=PromptDir,
            promptPrefix :str=PromptPrefix) -> dict:
        filepaths = glob.glob(os.path.join(directory, f"{promptPrefix}*"))
        filePostfix = ".txt"
        results = {}
        for filepath in filepaths:
            file_name = os.path.basename(filepath)
            with open(filepath, 'r') as file:
                file_name = file_name[len(promptPrefix):]
                if file_name.endswith(filePostfix):
                    file_name = file_name[:-len(filePostfix)]

                prompts = []
                content = ""
                for line in file.readlines():
                    if len(line.strip().replace(Loader.Sep, "")) == 0:
                        if len(content) > 0:
                            prompts.append(Prompt(content))
                            content = ""
                    else:
                        content += line

                results[file_name] = prompts
        return results

GLoader = Loader()