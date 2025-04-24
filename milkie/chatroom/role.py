from milkie.agent.agent import Agent
from milkie.agent.base_block import BaseBlock
from milkie.context import Context
from milkie.config.config import GlobalConfig
from milkie.response import Response

class Role(BaseBlock):
    def __init__(
            self, 
            name: str, 
            desc: str, 
            agent: Agent,
            context: Context, 
            config: GlobalConfig):
        super().__init__(context, config)

        self.name = name
        self.desc = desc
        self.agent = agent

    def compile(self):
        pass

    def execute(
            self, 
            context: Context, 
            query: str,
            args: dict, 
            prevBlock: BaseBlock=None, 
            **kwargs) -> Response:
        return self.agent.execute(
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)
