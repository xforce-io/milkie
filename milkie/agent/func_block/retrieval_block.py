from llama_index.core.schema import TextNode

from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.memory.memory_with_index import MemoryWithIndex
from milkie.response import Response
from milkie.retrieval.retrieval import RetrievalModule
from milkie.runtime.datasource import DataSource

class RetrievalBlock(FuncBlock):

    def __init__(
            self,
            context :Context = None,
            config :str = None,
            repoFuncs=None):
        super().__init__(
            agentName="Retrieval", 
            context=context, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "Retrieval"
        self.params = ["query"]
        
        if self.config.memoryConfig and self.config.indexConfig:
            self.memoryWithIndex = MemoryWithIndex(
                self.context.globalContext.settings,
                self.config.memoryConfig,
                self.config.indexConfig,
                self.context.globalContext.serviceContext)
        else:
            self.memoryWithIndex = context.getGlobalContext().memoryWithIndex

        self.dataSource :DataSource = context.getEnv().getDataSource()

    def compile(self):
        if self.isCompiled:
            return

        self.dataSource.setMainRetriever(RetrievalModule(
            globalConfig=self.context.globalContext.globalConfig,
            retrievalConfig=self.config.retrievalConfig,
            memoryWithIndex=self.memoryWithIndex,
            context=self.context))

        self.isCompiled = True

    def execute(
            self, 
            context: Context, 
            query: str = None, 
            args: dict = None, 
            prevBlock: BaseBlock = None, 
            **kwargs) -> Response:
        BaseBlock.execute(self, context, query, args, prevBlock, **kwargs)

        self._restoreParams(args, self.params)
        return self._retrieve(query if query else args["query"], args)

    def _retrieve(self, query :str, args :dict) -> Response:
        self.dataSource.getMainRetriever().retrieve(self.context)
        retrievalResult = self.context.retrievalResult
        if retrievalResult is None:
            return None
        
        response = Response()
        curBlock = ""
        blocks = []
        for result in retrievalResult:
            curContent = self._getBlockFromNode(result.node)
            if len(curBlock) + len(curContent) < self.config.retrievalConfig.blockSize:
                curBlock += curContent
            else:
                if len(curBlock) != 0:
                    blocks += [curBlock]
                    curBlock = curContent   
                else:
                    blocks += [curContent]

        if len(curBlock) > 0:
            blocks += [curBlock]

        response.metadata = {
            "blocks": blocks,
            "curQuery" : self.context.getCurQuery().query}
        response.respStr = "\n".join(blocks)
        return response

    def _getBlockFromNode(self, node :TextNode) ->str:
        return node.get_text()