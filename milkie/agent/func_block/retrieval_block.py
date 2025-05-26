from llama_index.core.schema import TextNode

from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.ontology.docset.docset_with_index import DocsetWithIndex
from milkie.ontology.retrieval.retrieval import RetrievalModule
from milkie.response import Response
from milkie.runtime.datasource import DataSource

class RetrievalBlock(FuncBlock):

    def __init__(
            self,
            globalContext: GlobalContext,
            config :str = None,
            repoFuncs=None):
        super().__init__(
            agentName="Retrieval", 
            globalContext=globalContext, 
            config=config, 
            repoFuncs=repoFuncs)

        self.funcName = "Retrieval"
        self.params = ["query"]
        
        if self.config.docsetConfig and self.config.indexConfig:
            self.docsetWithIndex = DocsetWithIndex(
                globalContext.settings,
                self.config.docsetConfig,
                self.config.indexConfig,
                globalContext.serviceContext)
        else:
            self.docsetWithIndex = globalContext.docsetWithIndex

        self.dataSource :DataSource = globalContext.getEnv().getDataSource()

    def compile(self):
        if self.isCompiled:
            return

        self.dataSource.setMainRetriever(RetrievalModule(
            globalConfig=self.globalContext.globalConfig,
            retrievalConfig=self.config.retrievalConfig,
            docsetWithIndex=self.docsetWithIndex))

        self.isCompiled = True

    def execute(
            self, 
            context: Context, 
            query: str,
            args: dict = None, 
            prevBlock: BaseBlock = None, 
            **kwargs) -> Response:
        BaseBlock.execute(
            self, 
            context=context, 
            query=query, 
            args=args, 
            prevBlock=prevBlock, 
            **kwargs)

        self._restoreParams(args, self.params)
        return self._retrieve(query, args)

    def createFuncCall(self):
        newFuncCall = RetrievalBlock(
            globalContext=self.globalContext, 
            config=self.config, 
            repoFuncs=self.repoFuncs
        )
        return newFuncCall

    def _retrieve(self, query: str, args: dict) -> Response:
        self.dataSource.getMainRetriever().retrieve(self.context, query, **args)
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
            "curQuery" : query}
        response.respStr = "\n".join(blocks)
        return response

    def _getBlockFromNode(self, node :TextNode) ->str:
        return node.get_text()