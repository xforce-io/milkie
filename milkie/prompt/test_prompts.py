from llama_index.legacy.prompts.base import ChatPromptTemplate, SelectorPromptTemplate
from llama_index.legacy.prompts.base import PromptTemplate
from llama_index.legacy.prompts.prompt_type import PromptType
from llama_index.legacy.prompts.utils import is_chat_model
from llama_index.legacy.llms.types import ChatMessage, MessageRole

from milkie.prompt.prompt import Loader

def candidateTextQAPromptImpl(promptQa:str):
    return PromptTemplate(
        Loader.load(promptQa), prompt_type=PromptType.QUERY_KEYWORD_EXTRACT
    )

def candidateRefinePromptImpl(promptRefine:str):
    return PromptTemplate(
        Loader.load(promptRefine), prompt_type=PromptType.REFINE
    )

def candidateTextQAPromptSel(
        promptSystem:str,
        promptQa:str) :
    return SelectorPromptTemplate(
        default_template=candidateTextQAPromptImpl(promptQa),
        conditionals=[(
            is_chat_model, 
            ChatPromptTemplate(
                message_templates=[
                    ChatMessage(
                        content=(Loader.load(promptSystem)),
                        role=MessageRole.SYSTEM,
                    ),
                    ChatMessage(
                        content=(Loader.load(promptQa)),
                        role=MessageRole.USER,
                    ),
            ])
        )]
    )

def candidateRefinePromptSel(promptRefine :str) :
    return SelectorPromptTemplate(
        default_template=candidateRefinePromptImpl(promptRefine),
        conditionals=[(
            is_chat_model, 
            ChatPromptTemplate(
                message_templates=[
                    ChatMessage(
                        content=(Loader.load(promptRefine)),
                        role=MessageRole.USER,
                    ),
            ])
        )]
    ) 