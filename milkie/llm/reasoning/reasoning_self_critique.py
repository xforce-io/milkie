from __future__ import annotations

from llama_index_client import ChatMessage
from milkie.global_context import GlobalContext
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.step_llm import StepLLM
from milkie.trace import stdout


class StepLLMCritique(StepLLM):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext, llm=None)

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs: dict) -> str:
        return f'''
            你是一个批判者，请对以下内容进行批评和修改：
            
            <<<原始指令>>>
            {args["instruction"]}

            <<<初始答案>>>
            {args["initialResponse"]}

            请直接给出你的修改意见(不是给出修改后的结果)：
        '''

class StepLLMRefine(StepLLM):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext, llm=None)

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs: dict) -> str:
        return f'''
            你是一个助手，请根据以下信息生成最终答案：
            
            <<<指令>>>
            {args["instruction"]}

            <<<初始答案>>>
            {args["initialResponse"]}

            <<<修改意见>>>
            {args["critiqueResponse"]}

            请直接生成最终答案：
        '''

class ReasoningSelfCritique(Reasoning):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            critiqueLLM: EnhancedLLM, 
            nIter: int = 3):
        super().__init__()
        self.globalContext = globalContext
        self.critiqueLLM = critiqueLLM
        self.nIter = nIter
        self.stepLLMCritique = StepLLMCritique(globalContext)
        self.stepLLMRefine = StepLLMRefine(globalContext)

    def reason(
            self, 
            llm: EnhancedLLM, 
            messages: list[ChatMessage], 
            stream: bool = False, 
            **kwargs) -> str:
        kwargs["temperature"] = 0.5
        
        # (1) 生成初始答案
        stdout(f"\n(1) init result", info=True, **kwargs)
        initialResponse = self._chat(
            llm=llm, 
            messages=messages, 
            stream=stream, 
            **kwargs)
        initialResponse = self._makeResp(
            resp=initialResponse, 
            stream=stream, 
            **kwargs)
        
        # (2) 生成修改意见
        stdout(f"\n(2) critique", info=True, **kwargs)
        args = {
            "instruction": messages[-1].content,
            "initialResponse": initialResponse
        }
        critiqueResponse = self.stepLLMCritique.llmCall(
            llm=self.critiqueLLM, 
            args=args, 
            stream=stream, 
            **kwargs)
        critiqueResponse = self._makeResp(
            resp=critiqueResponse, 
            stream=stream, 
            **kwargs)

        # (3) refine
        stdout(f"\n(3) refine", info=True, **kwargs)
        args = {
            "instruction": messages[-1].content,
            "initialResponse": initialResponse,
            "critiqueResponse": critiqueResponse
        }
        return self.stepLLMRefine.llmCall(
            llm=llm, 
            args=args, 
            stream=stream, 
            **kwargs)
