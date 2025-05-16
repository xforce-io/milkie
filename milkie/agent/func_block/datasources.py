from milkie.agent.base_block import BaseBlock
from milkie.agent.func_block.func_block import FuncBlock
from milkie.context import Context
from milkie.response import Response

class DataSources(FuncBlock):
    def __init__(self, globalContext, config, repoFuncs=None):
        super().__init__(
            agentName="DataSource",
            globalContext=globalContext,
            config=config,
            repoFuncs=repoFuncs
        )
        self.funcName = "DataSource"
        self.params = ["concepts"]
        
    def execute(self, context: Context, query: str, args: dict, **kwargs):
        BaseBlock.execute(self, context, query, args, **kwargs)

        self._restoreParams(args, self.params)
        concepts = args["concepts"]
        dataSources = self.globalContext.ontologyManager.getDataSourcesFromConcepts(concepts)
        dedupedDataSources = []
        for dataSource in dataSources:
            if dataSource not in dedupedDataSources:
                dedupedDataSources.append(dataSource)
                
        self.context.genResp(dedupedDataSources, **kwargs)
        return Response.buildFrom(dedupedDataSources)
    
    def createFuncCall(self):
        newFuncCall = DataSources(
            globalContext=self.globalContext,
            config=self.config,
            repoFuncs=self.repoFuncs)
        return newFuncCall
