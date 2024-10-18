from abc import ABC, abstractmethod
from milkie.global_context import GlobalContext
from milkie.llm.inference import chat
from milkie.response import Response

class StepLLM(ABC):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            promptMaker, 
            codeModel: bool = False):
        self.globalContext = globalContext
        self.promptMaker = promptMaker
        self.codeModel = codeModel
    
    def run(self, args: dict = {}, **kwargs):
        return self.formatResult(self.llmCall(args, **kwargs))
    
    @abstractmethod
    def makePrompt(self, useTool: bool = False, **args) -> str:
        pass

    def llmCall(self, args: dict, **kwargs) -> Response:
        self.prompt = self.makePrompt(useTool="tools" in kwargs, **args)
        llm = self.globalContext.settings.llmCode if self.codeModel else self.globalContext.settings.llm

        if "system_prompt" in args:
            systemPrompt = args["system_prompt"]
        else:
            systemPrompt = self.globalContext.globalConfig.getLLMConfig().systemPrompt

        return chat(
            llm=llm, 
            systemPrompt=systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            **kwargs)

    @abstractmethod
    def formatResult(self, result: Response):
        return result