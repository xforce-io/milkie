import logging
from llama_index.core import Response
from milkie.agent.base_agent import BaseAgent
from milkie.agent.prompt_agent import PromptAgent
from milkie.agent.retrieval_agent import RetrievalAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class FileLookupAgent(BaseAgent):
    
    def __init__(
            self,
            context :Context) -> None:
        super().__init__(context)
        
        self.retrievalAgent = RetrievalAgent(
            context)

        self.lookuper = PromptAgent(
            context, 
            "file_lookup")

    def execute(self, query: str, args: dict, **kwargs) -> Response:
        response = self.retrievalAgent.execute(args["query_str"])
        curQuery = response.metadata["curQuery"]
        blocks = response.metadata["blocks"]
        logger.debug(f"retrieval query[{curQuery}] result[{response}] blocks[{len(blocks)}]")
        
        return self.lookuper.execute(
            curQuery,
            args={
                "query_str":curQuery,
                "context_str":"\n".join(blocks)
            })

if __name__ == "__main__":
    filelookupAgent = FileLookupAgent(
        Context.createContext("config/global_filelookup.yaml"))
    response = filelookupAgent.execute(
        query="有个技术白皮书文档，路径是什么", 
        args={})
    print(response)