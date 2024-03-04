from llama_index import Response
from milkie.agent.prompt_agent import PromptAgent
from milkie.agent.retrieval_agent import RetrievalAgent
from milkie.agent.team.team import Team
from milkie.context import Context


class MapReduceQA(Team):

    def __init__(
            self, 
            context: Context, 
            config: str) -> None:
        super().__init__(context, config)

        self.retrievalAgent = RetrievalAgent(
            context, 
            config)

        self.blockQA = PromptAgent(
            context, 
            "qa_init")

        self.blockSummary = PromptAgent(
            context, 
            "block_summary")

    def task(self, query) -> Response:
        response = self.retrievalAgent.task(query)
        resps = []
        for block in response.metadata["blocks"]:
            resp = self.blockQA.task(
                query,
                query_str=query,
                context_str=block)
            resps.append(resp.response)

        sep = "\n-------------------\n"
        resp = self.blockSummary.task(
            query,
            query_str=query,
            blocks=sep.join(resps))
        return resp.response