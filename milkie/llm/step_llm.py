from abc import ABC, abstractmethod
from typing import Optional
from milkie.global_context import GlobalContext
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.response import Response
from milkie.llm.inference import chat, failChat
from milkie.trace import stdout
from milkie.utils.commons import getToolsSchemaForTools
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.llm.reasoning.reasoning_naive import ReasoningNaive

class StepLLM(ABC):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            promptMaker,
            llm :EnhancedLLM,
            reasoning: Reasoning = ReasoningNaive()):
        self.globalContext = globalContext
        self.promptMaker = promptMaker
        self.llm = llm
        self.reasoning = reasoning
    
    def completion(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> Response:
        return self.llmCall(
            llm=llm if llm else self.llm, 
            reasoning=reasoning if reasoning else self.reasoning,
            args=args, 
            stream=False, 
            **kwargs)
    
    def stream(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> Response:
        return self.llmCall(
            llm=llm if llm else self.llm, 
            reasoning=reasoning if reasoning else self.reasoning,
            args=args, 
            stream=True, 
            **kwargs)

    def streamAndFormat(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> Response:
        return self.formatResult(
            self.stream(llm, reasoning, args, **kwargs), 
            **kwargs)

    def streamOutputAndReturn(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> str:
        resp = self.llmCall(
            llm=llm if llm else self.llm, 
            reasoning=reasoning if reasoning else self.reasoning,
            args=args, 
            stream=True, 
            **kwargs)
        completeOutput = ""
        for chunk in resp.respGen:
            completeOutput += chunk.delta.content
            stdout(chunk.delta.content, end="", flush=True, **kwargs)
        return completeOutput

    def streamOutputAndFormat(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> Response:
        return self.formatResult(
            self.streamOutputAndReturn(llm, reasoning, args, **kwargs), 
            **kwargs)
    
    def completionAndFormat(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            **kwargs) -> Response:
        return self.formatResult(
            self.completion(llm=llm, reasoning=reasoning, args=args, **kwargs), 
            **kwargs)
    
    def makeSystemPrompt(self, args: dict, **kwargs) -> str:
        systemPrompt = None
        if "system_prompt" in args:
            systemPrompt = args["system_prompt"]

        if systemPrompt is None:
            systemPrompt = self.globalContext.settings.llmBasicConfig.systemPrompt
        return systemPrompt
    
    @abstractmethod
    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs: dict) -> str:
        pass

    def llmCall(
            self, 
            llm :EnhancedLLM, 
            reasoning: Optional[Reasoning] = None,
            args: dict = {}, 
            stream: bool = False, 
            **kwargs) -> Response:
        self.prompt = self.makePrompt(useTool="tools" in kwargs, args=args, **kwargs)
        self.systemPrompt = self.makeSystemPrompt(args=args, **kwargs)

        if "tools" in kwargs and kwargs["tools"] is not None and len(kwargs["tools"]) > 0:
            kwargs["tools"] = getToolsSchemaForTools(kwargs["tools"])
        else:
            kwargs.pop("tools", None)

        if reasoning is None:
            reasoning = self.reasoning

        return reasoning.reason(
            llm=llm, 
            systemPrompt=self.systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            stream=stream,
            **kwargs)

    def formatResult(self, result: Response, **kwargs):
        return result

    def fail(
            self, 
            llm :Optional[EnhancedLLM] = None, 
            **kwargs):
        failChat(
            llm=llm if llm else self.llm, 
            systemPrompt=self.systemPrompt,
            prompt=self.prompt, 
            promptArgs={},
            **kwargs)
