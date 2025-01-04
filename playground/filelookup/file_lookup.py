import logging
from milkie.agent.agents.base_agent import BaseAgent
from milkie.agent.llm_block.llm_block import LLMBlock
from milkie.agent.func_block.retrieval_block import RetrievalBlock
from milkie.context import Context
from milkie.response import Response

logger = logging.getLogger(__name__)

class FileLookupAgent(BaseAgent):
    
    def __init__(
            self,
            context :Context) -> None:
        super().__init__(context)
        
        self.retrievalAgent = RetrievalBlock(
            context)

        self.lookuper = LLMBlock(
            context, 
            "file_lookup")

    def execute(self, query: str, args: dict, **kwargs) -> Response:
        response = self.retrievalAgent.execute(args["query"])
        curQuery = response.metadata["curQuery"]
        blocks = response.metadata["blocks"]
        logger.debug(f"retrieval query[{curQuery}] result[{response}] blocks[{len(blocks)}]")
        
        return self.lookuper.execute(
            curQuery,
            args={
                "query":curQuery,
                "context":"\n".join(blocks)
            })

if __name__ == "__main__":
    filelookupAgent = FileLookupAgent(
        Context.create("config/global_filelookup.yaml"))
    response = filelookupAgent.execute(
        query="有个技术白皮书文档，路径是什么", 
        args={})
    print(response)