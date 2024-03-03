from abc import abstractmethod

from llama_index import ChatPromptTemplate, Response
from llama_index.llms.types import ChatMessage, MessageRole

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.prompt.prompt import Loader


class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        super.__init__(context, config)

        self.prompt = Loader.load(config)

    def task(self, query, **kwargs) -> Response:
        response = Response(response="", source_nodes=None)
        chatPromptTmpl = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=self.prompt,
                    role=MessageRole.USER)
            ]
        )
        response.response = self.context.globalContext.settings.llm.predict(
            prompt=chatPromptTmpl,
            query_str=query,
            kwargs=kwargs
        )
        return response