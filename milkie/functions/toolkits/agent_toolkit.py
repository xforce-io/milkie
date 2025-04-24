from milkie.functions.openai_function import OpenAIFunction
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.trace import stdout

class AgentToolkit(Toolkit):
    def __init__(self, agent):
        super().__init__()
        if agent is None:
            raise ValueError("Agent cannot be None")
        self.agent = agent
        self.agentFunction = self._createAgentFunction()

    def getName(self) -> str:
        if self.agent is None:
            raise ValueError("Agent is None")
        return self.agent.name

    def getDesc(self) -> str:
        if self.agent is None:
            raise ValueError("Agent is None")
        return self.agent.desc

    def execute(self, query :str, args :dict = {}, **kwargs):
        kwargs["top"] = False
        stdout(f"\n", info=True)
        return str(self.agent.execute(
            query=query, 
            args=args, 
            **kwargs))

    def _createAgentFunction(self):
        agentSchema = {
            "type": "function",
            "function": {
                "name": self.agent.name,
                "description": self.agent.desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to be processed by the agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        return OpenAIFunction(self.execute, agentSchema)

    def getTools(self):
        return [self.agentFunction]
