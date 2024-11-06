import logging
from typing import Any, Dict, Optional
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from milkie.context import History
from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.log import DEBUG
from milkie.response import Response

logger = logging.getLogger(__name__)

def chat(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        promptArgs :dict, 
        stream :bool = False,
        **kwargs) -> Response:
    if stream:
        return chatStream(llm, systemPrompt, prompt, promptArgs, **kwargs)
    else:
        return chatCompletion(llm, systemPrompt, prompt, promptArgs, **kwargs)

def chatCompletion(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        promptArgs :dict, 
        **kwargs) -> Response:
    prompt = preprocessPrompt(prompt)
    chatPromptTmpl = makeMessageTemplates(
        systemPrompt, 
        kwargs["history"] if "history" in kwargs else None, 
        prompt)
    response = Response(respStr="", source_nodes=None, metadata={})

    import time
    t0 = time.time()
    response.respStr, numTokens, chatCompletion = llm.predict(
        prompt=chatPromptTmpl,
        promptArgs=promptArgs,
        **kwargs)
    t1 = time.time()
    answer = response.respStr.replace("\n", "//")
    response.metadata["numTokens"] = numTokens
    response.metadata["chatCompletion"] = chatCompletion
    DEBUG(logger, f"chat prompt[{chatPromptTmpl.get_template()}] answer[{answer}] ({t1-t0:.2f}s)")
    return response

def preprocessPrompt(prompt :str) :
    # to prevent 'format' exception in get_template_vars
    return prompt.replace("{", "{{").replace("}", "}}")

def chatStream(
        llm: EnhancedLLM,
        systemPrompt: str,
        prompt: str,
        promptArgs: Dict[str, Any],
        **kwargs
) -> Response:
    prompt = preprocessPrompt(prompt)
    chatPromptTmpl = makeMessageTemplates(
        systemPrompt, 
        kwargs["history"] if "history" in kwargs else None, 
        prompt)
    response = Response(respGen=None, source_nodes=None, metadata={})

    import time
    t0 = time.time()
    response.respGen = llm.stream(prompt=chatPromptTmpl, promptArgs=promptArgs, **kwargs)
    t1 = time.time()

    DEBUG(logger, f"stream chat prompt[{chatPromptTmpl.get_template()}] ({t1-t0:.2f}s)")
    return response

def makeMessageTemplates(systemPrompt :str, history :Optional[History], prompt :str) -> ChatPromptTemplate:
    messageTemplates = []
    if history and history.use():
        history.setSystemPrompt(systemPrompt)
        history.addHistoryUserPrompt(prompt)
        messageTemplates = history.getDialogue()
    else:
        if systemPrompt:
            messageTemplates += [ChatMessage(content=systemPrompt, role=MessageRole.SYSTEM)]    
        messageTemplates += [ChatMessage(content=prompt, role=MessageRole.USER)]
    return ChatPromptTemplate(message_templates=messageTemplates)