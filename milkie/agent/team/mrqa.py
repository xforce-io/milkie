from llama_index.core import Response
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
            prompt="qa_init")

        self.blockProcess = PromptAgent(
            context, 
            prompt="block_process")

    def execute(self, input :str, task :str) -> Response:
        response = self.retrievalAgent.execute(input)
        resps = []
        maxCnt = 10
        cnt = 0
        for block in response.metadata["blocks"]:
            resp = self.blockQA.execute(
                input,
                query_str=task,
                context_str=block)
            resps.append(resp.response)
            cnt += 1
            if cnt >= maxCnt:
                break

        sep = "\n-------------------\n"
        resp = self.blockProcess.execute(
            input,
            query_str=task,
            blocks=sep.join(resps))
        return resp

if __name__ == "__main__":
    agent = MapReduceQA()
    response = agent.execute(
        input="@filepath:有个技术白皮书文档，路径是什么", 
        task="写一份总结，不超过 100 字")
    print(response) 