import time
from milkie.agent.memory.history import History
from milkie.agent.memory.storage.storage import Storage
from milkie.config.config import StorageConfig
from milkie.global_context import GlobalContext


class BaseMemory(object):
    def __init__(
            self, 
            type :str,
            storageConfig :StorageConfig, 
            globalContext :GlobalContext):
        self.type = type
        self.storageConfig = storageConfig
        self.globalContext = globalContext
        self.storage = Storage(type, storageConfig)
        self.lastFlushTime = 0

    def flush(self, mem :list[dict]):
        if time.time() - self.lastFlushTime > self.storageConfig.flushIntervalSec:
            self.storage.flush(mem)
            self.lastFlushTime = time.time()

    def load(self):
        return self.storage.load()
        