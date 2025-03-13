from llama_index_client import ChatMessage
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.llm.inference import chat
from milkie.llm.reasoning.reasoning import Reasoning

class ReasoningNaive(Reasoning):

    def reason(
            self, 
            llm: EnhancedLLM, 
            messages: list[ChatMessage], 
            stream: bool = False,
            **kwargs) -> str:
        return self._chat(
            llm=llm, 
            messages=messages, 
            stream=stream, 
            **kwargs)
