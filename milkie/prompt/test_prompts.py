from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

from milkie.prompt.prompt import Loader

EMPH_QUERY_REFINE_PROMPT_TMPL = Loader.load("qa_refine")

EMPH_QUERY_REFINE_PROMPT = PromptTemplate(
    EMPH_QUERY_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

CANDIDATE_TEXT_QA_PROMPT_IMPL = DEFAULT_TEXT_QA_PROMPT
CANDIDATE_REFINE_PROMPT_IMPL = EMPH_QUERY_REFINE_PROMPT