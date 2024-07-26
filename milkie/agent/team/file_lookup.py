from llama_index.core import Response
from milkie.agent.base_agent import BaseAgent
from milkie.agent.prompt_agent import PromptAgent
from milkie.agent.retrieval_agent import RetrievalAgent
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory

class FileLookupAgent(BaseAgent):
    
    def __init__(
            self,
            context :Context,
            config :str) -> None:
        super().__init__(context, config)
        
        self.retrievalAgent = RetrievalAgent(
            context, 
            config)

        self.lookuper = PromptAgent(
            context, 
            "file_lookup")

    def execute(self, query: str, args: dict, **kwargs) -> Response:
        response = self.retrievalAgent.execute(query)
        blocks = response.metadata["blocks"]
        return self.lookuper.execute(
            query,
            args={
                "query_str":query,
                "context_str":"\n".join(blocks)
            })

if __name__ == "__main__":
    globalConfig = GlobalConfig("config/global_filelookup.yaml")
    globalContext = GlobalContext(
        globalConfig, 
        ModelFactory())
    context = Context(globalContext)
    filelookupAgent = FileLookupAgent(
        context, 
        config="qa")
    response = filelookupAgent.execute(
        query="有个技术白皮书文档，路径是什么", 
        args={})
    print(response)