from milkie.agent.memory.storage.storage_device import StorageDevice
from milkie.config.config import StorageConfig
from abc import abstractmethod

class Storage(object):
    def __init__(
            self, 
            type :str,
            config :StorageConfig):
        self.type = type
        self.config = config
        self.storageDevice = StorageDevice(config, type)

    def load(self) -> dict:
        return self.storageDevice.load()

    def flush(self, mem :dict):
        self.storageDevice.flush(mem)
