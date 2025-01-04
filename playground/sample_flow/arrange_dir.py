from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.filesys_toolkit import FilesysToolkit
from milkie.response import Response

class ArrangeDir(BaseAgent):
    
    def __init__(self, context: Context = None, config: str | GlobalConfig = None) -> None:
        super().__init__(context, config)

        flowCode = """
                0. 获取目录{dir}下的树形结构 -> fileTree
                1. 根据{fileTree}，分析出一些主题或者分类,请直接输出 -> categories
                2. #THOUGHT 使用Toolkit中工具将目录{dir}下的所有文件，按照主题整理后放置在目录{destDir}中(主题为[{categories}])。
                    注意：请不要改变{dir}下的任何文件
                    {dir}文件树如下： --{fileTree}--
                3. #DECOMPOSE
            """

        self.flowBlock = FlowBlock(
            flowCode=flowCode, 
            toolkit=FilesysToolkit(), 
            usePrevResult=False)
        self.flowBlock.compile()

    def execute(self, args: dict, **kwargs) -> Response:
        return self.flowBlock.execute(args=args)
    
if __name__ == "__main__":
    arrangeDir = ArrangeDir()

    args = {
        "dir" : "/Users/xupeng/Documents/aishu/test/",
        "destDir" : "/Users/xupeng/Documents/aishu/test_bak",
    }
    arrangeDir.execute(args=args)
