import json
import os

from milkie.config.config import StorageConfig


class StorageDevice(object):
    def __init__(
            self, 
            config :StorageConfig,
            type :str):
        self.config = config
        self.type = type

    def flush(self, data :dict):
        if not os.path.exists(self.config.mempath):
            os.makedirs(self.config.mempath)
            
        with open(self.getFilePath(), "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> dict:
        if not os.path.exists(self.getFilePath()):
            return []
        
        with open(self.getFilePath(), "r", encoding='utf-8') as f:
            return json.load(f)
    
    def build(self, config :StorageConfig) -> 'StorageDevice':
        return StorageDevice(config, self.type)

    def getFilePath(self) -> str:
        return os.path.join(self.config.mempath, f"{self.type}.json")