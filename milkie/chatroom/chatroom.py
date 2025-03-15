import logging

from milkie.agent.agent import Agent
from milkie.agent.base_block import BaseBlock
from milkie.chatroom.role import Role
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.response import Response
from milkie.trace import stdout

logger = logging.getLogger(__name__)

class Chatroom(BaseBlock):
    def __init__(
            self, 
            name: str,
            desc: str,
            host: str,
            globalContext: GlobalContext, 
            config: GlobalConfig):
        super().__init__(globalContext, config)

        self.name = name
        self.desc = desc
        self.host = host
        self.roles = dict[str, Role]()
        self.onlyRole :Role = None
        self.prologue = None

    def setHost(self, agentMap: dict[str, Agent]):
        if isinstance(self.host, str):
            self.host = Role(
                name=self.host, 
                desc=self.host, 
                agent=agentMap[self.host],
                context=self.context, 
                config=self.config)

    def assignRole(self, agent: Agent, roleName: str):
        self.roles[roleName] = Role(
            name=roleName, 
            desc=roleName, 
            agent=agent,
            context=self.context, 
            config=self.config)
        self.onlyRole = self.roles[roleName]

    def compile(self):
        for name, role in self.roles.items():
            role.setContext(self.context)

    def execute(
            self, 
            context: Context, 
            query: str,
            args: dict, 
            **kwargs) -> Response:
        if len(self.roles) == 1:
            return self.execute1v1Mode(context, query, args, **kwargs)
        else:
            raise NotImplementedError("Multi-agent mode is not implemented yet.")
    
    def execute1v1Mode(self, context: Context, query: str, args: dict, **kwargs) -> Response:
        stdout(f"\nROLE[{self.host.name}] => \n{query}", info=True)
        self.host.context.addHistoryAssistantPrompt(query)
        while True: 
            stdout(f"\n\nROLE[{self.onlyRole.name}] => ", info=True)
            self.onlyRole.context.addHistoryUserPrompt(query)
            resp = self.onlyRole.execute(context, args, **kwargs)
            if resp.isEnd():
                break

            query = resp.respStr

            stdout(f"\n\nROLE[{self.host.name}] => ", info=True)
            self.host.context.addHistoryUserPrompt(query)
            resp = self.host.execute(context, args, **kwargs)
            if resp.isEnd():
                break

            query = resp.respStr
        return resp
