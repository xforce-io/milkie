import logging
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.llm.enhanced_llm import EnhancedLLM
from milkie.log import DEBUG
from milkie.response import Response

logger = logging.getLogger(__name__)

def chat(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        promptArgs :dict, 
        **kwargs) -> Response:
    prompt = preprocessPrompt(prompt)

    response = Response(respStr="", source_nodes=None, metadata={})

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

def chatBatch(
        llm :EnhancedLLM, 
        systemPrompt :str,
        prompt :str, 
        argsList :list[dict], 
        **kwargs) -> list[Response]:
    prompt = preprocessPrompt(prompt)

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

    resultBatch = llm.predictBatch(
        prompt=chatPromptTmpl,
        argsList=argsList,
        **kwargs)

    responses = []
    for result in resultBatch:
        response = Response(respStr=result[0], source_nodes=None, metadata={})
        response.metadata["numTokens"] = result[1]
        response.metadata["chatCompletion"] = result[2]
        responses += [response]
    return responses

def preprocessPrompt(prompt :str) :
    # to prevent 'format' exception in get_template_vars
    return prompt.replace("{", "{{").replace("}", "}}")