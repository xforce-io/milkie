import logging

from llama_index.core import Response

from milkie.agent.base_agent import BaseAgent
from milkie.context import Context
from milkie.llm.inference import chat, chatBatch
from milkie.prompt.prompt import Loader

logger = logging.getLogger(__name__)

class PromptAgent(BaseAgent):

    def __init__(
            self,
            context :Context = None,
            config :str = None,
            prompt :str = None) -> None:
        super().__init__(context, config)

        prompt = prompt if prompt else config["prompt"]
        self.prompt = Loader.load(prompt) if prompt else None

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

if __name__ == "__main__":
    agent = PromptAgent()
    agent.execute("你好", args={})