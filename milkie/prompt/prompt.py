
DirPrompt = "prompts/"

class Prompt:
    def __init__(self, content :str):
        self.content = content

    def getContent(self) -> str:
        return self.content

class Loader:
    def load(name :str) -> str:
        pathPrompt = f"{DirPrompt}{name}.txt"
        with open(pathPrompt, 'r') as file:
            content = file.read()
        return content

GLoader = Loader()