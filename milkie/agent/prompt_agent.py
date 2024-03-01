from abc import abstractmethod

from llama_index import PromptTemplate, Response
from llama_index.llms.types import ChatMessage, MessageRole

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.prompt.prompt import GLoader


class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        self.__init__(context, config)

        self.prompt = GLoader.load(config)

    def task(self, query, **kwargs) -> Response:
        response = Response()
        messages = []
        messages.append(ChatMessage(
            content=self.prompt,
            role=MessageRole.USER))
        response.response = self.context.settings.llm.chat(
            messages=messages,
            query_str=query,
            kwargs=kwargs
        )
        return response