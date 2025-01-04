from milkie.global_context import GlobalContext
from milkie.llm.reasoning.reasoning import Reasoning
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.step_llm import StepLLM
from milkie.trace import stdout

class StepLLMReasoningSelfConsistency(StepLLM):
    def __init__(
            self, 
            globalContext: GlobalContext):
        super().__init__(
            globalContext, 
            promptMaker=None, 
            llm=None)

    def makePrompt(
            self, 
            useTool: bool = False, 
            args: dict = {}, 
            **kwargs: dict) -> str:
        return f'''
            你是一个助手，需要根据指令及多个候选答案总结最终的答案，要求如下：
            （1）尽可能选取多数人的结论
            
            <<<指令>>>
            {args["instruction"]}

            <<<候选答案>>>
            {args["candidates"]}

            <<<最终答案>>>
        '''

class ReasoningSelfConsistency(Reasoning):

    def __init__(
            self, 
            globalContext: GlobalContext,
            amateur: EnhancedLLM = None,
            nIter: int = 3):
        self.amateur = amateur
        self.nIter = nIter
        self.stepLLM = StepLLMReasoningSelfConsistency(globalContext)

    def reason(
            self, 
            llm: EnhancedLLM, 
            systemPrompt: str, 
            prompt: str, 
            promptArgs: dict, 
            stream: bool = False,
            **kwargs) -> str:
        resps = []
        kwargs["temperature"] = 0.5
        for i in range(self.nIter):
            resp = self._chat(
                llm=self.amateur if self.amateur else llm, 
                systemPrompt=systemPrompt, 
                prompt=f"trial {i}: {prompt}", 
                promptArgs=promptArgs, 
                stream=stream,
                **kwargs)
            resps.append(resp)

        candidates = []
        cnt = 0 
        for resp in resps:
            stdout(f"\ntrial {cnt}", info=True, **kwargs)
            candidates.append(self._makeResp(resp, stream=stream, **kwargs))
            cnt += 1
        candidates = "\n".join(candidates)

        args = {
            "instruction": prompt,
            "candidates": candidates
        }
        stdout(f"\nfinal", info=True, **kwargs)
        return self.stepLLM.llmCall(
            llm=llm, 
            args=args, 
            stream=stream, 
            **kwargs)