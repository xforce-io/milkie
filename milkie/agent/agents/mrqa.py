from milkie.agent.llm_block import LLMBlock
from milkie.agent.retrieval_block import RetrievalBlock
from milkie.agent.agents.base_agent import BaseAgent
from milkie.context import Context
from milkie.response import Response


class MapReduceQA(BaseAgent):

    def __init__(
            self, 
            context: Context=None, 
            config: str=None) -> None:
        super().__init__(context, config)

        self.retrievalAgent = RetrievalBlock(
            self.context, 
            self.config)

        self.blockQA = LLMBlock(
            self.context, 
            prompt="block_process")

        self.blockProcess = LLMBlock(
            self.context, 
            prompt="block_process")

    def execute(self, input :str, task :str) -> Response:
        response = self.retrievalAgent.execute(input)
        if response is None:
            return None
        
        resps = []
        maxCnt = 50
        cnt = 0
        for block in response.metadata["blocks"]:
            resp = self.blockQA.execute(
                input,
                args={
                    "query":task,
                    "blocks":block})
            resps.append(resp.response)
            cnt += 1
            if cnt >= maxCnt:
                break

        sep = ""
        resp = self.blockProcess.execute(
            input,
            args={
                "query":task,
                "blocks":sep.join(resps)})
        return resp

if __name__ == "__main__":
    agent = MapReduceQA()
    response = agent.execute(
        input="@filepath:/Users/xupeng/Documents/eb/materials/knowledge/知识图谱构建技术综述.pdf", 
        task="写一份总结，不超过 300 字")
    print(response) 