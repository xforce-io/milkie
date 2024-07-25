import logging
from llama_index.core import Response, ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.llm.enhanced_llm import EnhancedLLM

logger = logging.getLogger(__name__)

def chat(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        promptArgs :dict, 
        **kwargs) -> Response:
    response = Response(response="", source_nodes=None, metadata={})

    messageTemplates = []
    if systemPrompt:
        messageTemplates += [
            ChatMessage(
                content=systemPrompt,
                role=MessageRole.SYSTEM)
        ]
    
    messageTemplates += [
        ChatMessage(
            content=prompt,
            role=MessageRole.USER)
    ]

    chatPromptTmpl = ChatPromptTemplate(message_templates=messageTemplates)

    import time
    t0 = time.time()
    response.response, numTokens = llm.predict(
        prompt=chatPromptTmpl,
        promptArgs=promptArgs,
        **kwargs)
    t1 = time.time()
    answer = response.response.replace("\n", "//")
    response.metadata["numTokens"] = numTokens
    logger.debug(f"chat prompt[{prompt}] answer[{answer}] ({t1-t0:.2f}s)")
    return response

def chatBatch(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        argsList :list[dict], 
        **kwargs) -> list[Response]:
    messageTemplates = []
    if systemPrompt:
        messageTemplates += [
            ChatMessage(
                content=systemPrompt,
                role=MessageRole.SYSTEM)
        ]
    
    messageTemplates += [
        ChatMessage(
            content=prompt,
            role=MessageRole.USER)
    ]

    chatPromptTmpl = ChatPromptTemplate(message_templates=messageTemplates)

    import time
    t0 = time.time()
    resultBatch = llm.predictBatch(
        prompt=chatPromptTmpl,
        argsList=argsList,
        **kwargs)
    t1 = time.time()

    responses = []
    for result in resultBatch:
        response = Response(response=result[0], source_nodes=None, metadata={})
        response.metadata["numTokens"] = result[1]
        responses += [response]
    return responses