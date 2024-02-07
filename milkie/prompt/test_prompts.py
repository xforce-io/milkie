from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

EMPH_QUERY_REFINE_PROMPT_TMPL = (
    "原始问题如下: {query_str}\n"
    "我们已经有了一个答案: {existing_answer}\n"
    "我们现在有机会让这个答案变得更好, "
    "（如果有必要的话) 根据下面的这些上下文信息.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "根据这些新的上下文，尝试让答案变得更好, 优化这个问题。如果这些新的上下文信息无助于改进答案，请返回原始回答。"
    "原始问题如下: {query_str}\n"
    "优化后的回答: "
)

EMPH_QUERY_REFINE_PROMPT = PromptTemplate(
    EMPH_QUERY_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

CANDIDATE_TEXT_QA_PROMPT_IMPL = DEFAULT_TEXT_QA_PROMPT
CANDIDATE_REFINE_PROMPT_IMPL = EMPH_QUERY_REFINE_PROMPT