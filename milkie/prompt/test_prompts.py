from llama_index import ChatPromptTemplate, SelectorPromptTemplate
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.utils import is_chat_model
from llama_index.llms.types import ChatMessage, MessageRole

from milkie.prompt.prompt import Loader

EMPH_QUERY_INIT_PROMPT_TMPL = Loader.load("qa_init")
EMPH_QUERY_REFINE_PROMPT_TMPL = Loader.load("qa_refine")

EMPH_QUERY_INIT_PROMPT = PromptTemplate(
    EMPH_QUERY_INIT_PROMPT_TMPL, prompt_type=PromptType.QUERY_KEYWORD_EXTRACT
)

EMPH_QUERY_REFINE_PROMPT = PromptTemplate(
    EMPH_QUERY_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

EMPH_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=EMPH_QUERY_INIT_PROMPT,
    conditionals=[(
        is_chat_model, 
        ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=(Loader.load("system_prompt")),
                    role=MessageRole.SYSTEM,
                ),
                ChatMessage(
                    content=(Loader.load("qa_init")),
                    role=MessageRole.USER,
                ),
        ])
    )]
)

EMPH_REFINE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=EMPH_QUERY_REFINE_PROMPT,
    conditionals=[(
        is_chat_model, 
        ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=(Loader.load("qa_refine")),
                    role=MessageRole.USER,
                ),
        ])
    )]
)

CANDIDATE_TEXT_QA_PROMPT_IMPL = EMPH_QUERY_INIT_PROMPT
CANDIDATE_REFINE_PROMPT_IMPL = EMPH_QUERY_REFINE_PROMPT
CANDIDATE_TEXT_QA_PROMPT_SEL = EMPH_TEXT_QA_PROMPT_SEL
CANDIDATE_REFINE_PROMPT_SEL = EMPH_REFINE_PROMPT_SEL