from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.flow_block import FlowBlock
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.functions.toolkits.filesys_toolkits import FilesysToolKits
from milkie.response import Response

class ArrangeDir(BaseAgent):
    
    def __init__(self, context: Context = None, config: str | GlobalConfig = None) -> None:
        super().__init__(context, config)

        flowCode = """
                0. 获取目录{dir}下的文件树 -> fileTree
                1. #THOUGHT 整理目录{dir}下的文件, 根据文件树{fileTree}
                2. #DECOMPOSE
            """

        self.flowBlock = FlowBlock(
            flowCode=flowCode, 
            toolkit=FilesysToolKits(), 
            usePrevResult=False)
        self.flowBlock.compile()

    def execute(self, args: dict, **kwargs) -> Response:
        return self.flowBlock.execute(args=args)
    
if __name__ == "__main__":
    arrangeDir = ArrangeDir()

    args = {
        "dir" : "/Users/xupeng/Documents/aishu/test/",
    }
    arrangeDir.execute(args=args)
