from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.toolkits.basic_toolkit import BasicToolKit
from milkie.response import Response

class Writer(BaseAgent):
    
    def __init__(
            self, 
            context: Context = None, 
            config: str | GlobalConfig = None,
            toolkit: Toolkit = None
        ) -> None:
        super().__init__(context, config)

        flowCode = """
                1. 详细分析下面的问题【{topic}】 -> topicInfo
                2. 我现在要针对主题【{topic}】写一篇文章，根据下述信息写一篇摘要【{topicInfo}】: -> summary
                3. 根据摘要【{summary}】写一篇两级大纲 => {{ "段落标题1/子段落标题1":"","段落标题1/子段落标题2":"","段落标题2/子段落标题1":"", ... }} -> outlines
                FOR outline in outlines :
                    1. 正在写主题为【{topic}】的文章，段落标题为：{outline.key}，请开始段落写作，不超过 500 字 -> paragraph
                    2. #RET {{ "{{outline.key}}" : "{{paragraph}}" }}
                END -> paragraphs
                4. 用 markdown 格式化下面内容:{paragraphs} -> markdown
                5. 邮件发送给{email}, 邮件标题为{topic}, 邮件内容为{{markdown}}
            """

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
    writer = Writer(context=context, toolkit=BasicToolKit(context.getGlobalContext()))
    args = {
        "topic" : "为什么中国队这么差",
        "email" : "freeman.xu@aishu.cn"
    }
    writer.execute(args=args)
