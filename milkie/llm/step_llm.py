import logging
from abc import ABC, abstractmethod
from milkie.context import GlobalContext
from milkie.llm.inference import chat
from llama_index.core import Response

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
    def makePrompt(self, **args) -> str:
        pass

    def llmCall(self, args: dict, **kwargs) -> Response:
        self.prompt = self.makePrompt(**args)
        llm = self.globalContext.settings.llmCode if self.codeModel else self.globalContext.settings.llm
        return chat(
            llm=llm, 
            systemPrompt=self.globalContext.globalConfig.getLLMConfig().systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            **kwargs)

    def formatResult(self, result: Response):
        pass