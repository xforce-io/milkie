from typing import Any, Callable, Optional, Sequence
from llama_index import BasePromptTemplate
from llama_index.llms.types import ChatMessage
from llama_index.llms import HuggingFaceLLM

class EnhancedHFLLM(HuggingFaceLLM) :

    def __init__(
            self, 
            context_window: int, 
            max_new_tokens: int, 
            query_wrapper_prompt: str, 
            tokenizer_name: str, 
            model_name: str, 
            tokenizer_kwargs: dict, 
            model_kwargs: dict, 
            generate_kwargs: dict, 
            is_chat_model: bool, 
            system_prompt: str, 
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]]) -> None:
        super().__init__(
            context_window=context_window, 
            max_new_tokens=max_new_tokens, 
            query_wrapper_prompt=query_wrapper_prompt, 
            tokenizer_name=tokenizer_name, 
            model_name=model_name, 
            tokenizer_kwargs=tokenizer_kwargs, 
            model_kwargs=model_kwargs, 
            generate_kwargs=generate_kwargs, 
            is_chat_model=is_chat_model, 
            system_prompt=system_prompt, 
            messages_to_prompt=messages_to_prompt)
    
    def predict(
            self, 
            prompt: BasePromptTemplate, 
            **prompt_args: Any) -> str:
        self._log_template_data(self, prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            response = self.chat(messages)
            output = response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt)
            output = response.text
        return (self._parse_output(output), len(response.raw["model_output"][0]))