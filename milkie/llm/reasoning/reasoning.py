from milkie.llm.enhanced_llm import EnhancedLLM
from abc import ABC, abstractmethod

from milkie.llm.inference import chat
from milkie.response import Response
from milkie.trace import stdout

class Reasoning(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def reason(
            self, 
            llm :EnhancedLLM, 
            systemPrompt :str,
            prompt :str, 
            promptArgs :dict,
            stream :bool = False,
            **kwargs) -> str:
        pass

    def _makeResp(
            self, 
            resp: Response, 
            output: bool = True,
            stream: bool = False,
            **kwargs):
        candidates = ""
        if stream:
            for chunk in resp.respGen:
                candidates += chunk.message.content
                if output:
                    stdout(chunk.message.content, info=True, end="", flush=True, **kwargs)
        else:
            if output:
                stdout(resp, info=True, flush=True, **kwargs)
            candidates = resp.resp
        return candidates

    def _chat(
            self, 
            llm: EnhancedLLM, 
            systemPrompt: str, 
            prompt: str, 
            promptArgs: dict, 
            stream: bool = False,
            **kwargs) -> str:
        return chat(
            llm=llm, 
            systemPrompt=systemPrompt, 
            prompt=prompt, 
            promptArgs=promptArgs, 
            stream=stream, 
            **kwargs)
