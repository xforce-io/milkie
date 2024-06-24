from llama_index.core import Response, ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.llm.enhanced_llm import EnhancedLLM

def chat(
        llm :EnhancedLLM, 
        prompt :str, 
        promptArgs :dict, 
        **kwargs) -> Response:
    response = Response(response="", source_nodes=None, metadata={})
    chatPromptTmpl = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content=prompt,
                role=MessageRole.USER)
        ]
    )

    import time
    t0 = time.time()
    response.response, numTokens = llm.predict(
        prompt=chatPromptTmpl,
        promptArgs=promptArgs,
        **kwargs)
    t1 = time.time()
    answer = response.response.replace("\n", "//")
    response.metadata["numTokens"] = numTokens
    return response

def chatBatch(
        llm :EnhancedLLM, 
        prompt :str, 
        argsList :list[dict], 
        **kwargs) -> list[Response]:
    chatPromptTmpl = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content=prompt,
                role=MessageRole.USER)
        ]
    )

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