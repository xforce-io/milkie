import logging

from llama_index.core import Response
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.llm.inference import chat, chatBatch
from milkie.prompt.prompt import Loader

logger = logging.getLogger(__name__)

class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context,
            config :str) -> None:
        super().__init__(context, config)

        self.prompt = Loader.load(config) if config else None

    def execute(self, query :str, args :dict, **kwargs) -> Response:
        return chat(
            llm=self.context.globalContext.settings.llm, 
            systemPrompt=None,
            prompt=self.prompt if self.prompt else query, 
            promptArgs=args, 
            **kwargs) 

    def executeBatch(self, query :str, argsList :list[dict], **kwargs) -> list[Response]:
        return chatBatch(
            llm=self.context.globalContext.settings.llm, 
            systemPrompt=None,
            prompt=self.prompt if self.prompt else query, 
            argsList=argsList, 
            **kwargs) 