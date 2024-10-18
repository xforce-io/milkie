from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.basic_toolkit import BasicToolKit
from milkie.response import Response

class TestAgent(BaseAgent):
    
    def __init__(
            self, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None
        ) -> None:
        super().__init__(context, config)

        flowCode = "邮件发送给{email}, 邮件标题为{topic}, 邮件内容为'ABC'"

        self.flowBlock = FlowBlock(
            flowCode=flowCode, 
            usePrevResult=False,
            toolkit=toolkit
        )
        self.flowBlock.compile()

    def execute(self, args: dict, **kwargs) -> Response:
        return self.flowBlock.execute(args=args)
    
if __name__ == "__main__":
    context = Context.create("config/global.yaml")
    writer = TestAgent(context=context, toolkit=BasicToolKit(context.getGlobalContext()))
    args = {
        "topic" : "为什么中国队这么差",
        "email" : "freeman.xu@aishu.cn"
    }
    writer.execute(args=args)
