from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.inference import chat
from milkie.llm.reasoning.reasoning import Reasoning

class ReasoningNaive(Reasoning):

    def reason(
            self, 
            llm: EnhancedLLM, 
            systemPrompt: str, 
            prompt: str, 
            promptArgs: dict, 
            stream: bool = False,
            **kwargs) -> str:
        return self._chat(
            llm=llm, 
            systemPrompt=systemPrompt, 
            prompt=prompt, 
            promptArgs=promptArgs, 
            stream=stream, 
            **kwargs)
