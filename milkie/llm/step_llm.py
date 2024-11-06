from abc import ABC, abstractmethod
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.llm.inference import chat
from milkie.trace import stdout
from milkie.utils.commons import getToolsSchemaForTools

class StepLLM(ABC):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            promptMaker, 
            codeModel: bool = False):
        self.globalContext = globalContext
        self.promptMaker = promptMaker
        self.codeModel = codeModel
    
    def completion(self, args: dict = {}, **kwargs):
        return self.llmCall(args, stream=False, **kwargs)
    
    def stream(self, args: dict = {}, **kwargs):
        return self.llmCall(args, stream=True, **kwargs)

    def streamAndFormat(self, args: dict = {}, **kwargs):
        return self.formatResult(self.stream(args, **kwargs), **kwargs)

    def streamOutputAndReturn(self, args: dict = {}, **kwargs):
        resp = self.llmCall(args, stream=True, **kwargs)
        completeOutput = ""
        for chunk in resp.respGen:
            completeOutput += chunk.delta.content
            stdout(chunk.delta.content, end="", flush=True, **kwargs)
        return completeOutput

    def streamOutputAndFormat(self, args: dict = {}, **kwargs):
        return self.formatResult(self.streamOutputAndReturn(args, **kwargs), **kwargs)
    
    def completionAndFormat(self, args: dict = {}, **kwargs):
        return self.formatResult(self.completion(args, **kwargs), **kwargs)
    
    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        systemPrompt = None
        if "system_prompt" in args:
            systemPrompt = args["system_prompt"]

        if systemPrompt is None:
            systemPrompt = self.globalContext.globalConfig.getLLMConfig().systemPrompt
        return systemPrompt
    
    @abstractmethod
    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs: dict) -> str:
        pass

    def llmCall(self, args: dict, stream: bool, **kwargs) -> Response:
        self.prompt = self.makePrompt(useTool="tools" in kwargs, args=args, **kwargs)
        llm = self.globalContext.settings.llmCode if self.codeModel else self.globalContext.settings.llm
        systemPrompt = self.makeSystemPrompt(args=args, **kwargs)

        if "tools" in kwargs and kwargs["tools"] is not None and len(kwargs["tools"]) > 0:
            kwargs["tools"] = getToolsSchemaForTools(kwargs["tools"])
        else:
            kwargs.pop("tools", None)

        return chat(
            llm=llm, 
            systemPrompt=systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            stream=stream,
            **kwargs)

    @abstractmethod
    def formatResult(self, result: Response, **kwargs):
        return result