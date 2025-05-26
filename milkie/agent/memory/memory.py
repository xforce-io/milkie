from milkie.agent.memory.experience import Experience
from milkie.agent.memory.history import History
from milkie.agent.memory.knowhow import Knowhow
from milkie.agent.memory.storage.storage import Storage
from milkie.config.config import MemoryConfig
from milkie.global_context import GlobalContext


class Memory(object):
    def __init__(
            self, 
            memoryConfig :MemoryConfig,
            globalContext :GlobalContext):
        self.memoryConfig = memoryConfig

        self.history = History()
        self.knowhow = Knowhow(
            storageConfig=memoryConfig.storageConfig,
            knowhowConfig=memoryConfig.knowhowConfig,
            globalContext=globalContext)

    def getHistory(self) -> History:
        return self.history
    
    def getKnowhow(self) -> Knowhow:
        return self.knowhow

    def extract(self, history :History):
        self.knowhow.extract(history)

    def addKnowhow(
            self, 
            key :str,
            value :str,
            score :int, 
            timestamp :int):
        self.storage.add(
            type = "knowhow",
            key = key,
            value = value,
            score = score,
            timestamp = timestamp
        )

    def addExperience(
            self, 
            key :str,
            value :str,
            score :int, 
            timestamp :int):
        self.storage.add(
            type = "experience",
            key = key,
            value = value,
            score = score,
            timestamp = timestamp
        )

