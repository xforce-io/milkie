from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.base_toolkits import BaseToolkit
from milkie.response import Response

class Cmdline(BaseAgent):
    def __init__(
        self, 
        context: Context = None, 
        config: str | GlobalConfig = None,
        flowCode: str = None,
        toolkit: BaseToolkit = None
    ) -> None:
        super().__init__(context, config)

        self.flowBlock = FlowBlock(
            flowCode=flowCode, 
            context=self.context,
            config=self.config,
            toolkit=toolkit, 
            usePrevResult=False)
        self.flowBlock.compile()

    def execute(self, args: dict, **kwargs) -> Response:
        return self.flowBlock.execute(args=args)
    
if __name__ == "__main__":
    pass
