from typing import List
from llama_index import SimpleDirectoryReader
from milkie.config.config import LongTermMemorySource, MemoryTermConfig


class Memory(object):
    def __init__(self, memoryTermConfigs :List[MemoryTermConfig]):
        self.docSet = []
        for memoryTermConfig in memoryTermConfigs:
            if memoryTermConfig.source == LongTermMemorySource.LOCAL.name:
                docs = self.__buildDocsFromLongTermLocal(memoryTermConfig)
                self.docSet.append(docs)
            else:
                raise Exception(f"Not supported long term memory type[{memoryTermConfig.source}]")
    
    def __buildDocsFromLongTermLocal(self, memoryTermConfig :MemoryTermConfig):
        return SimpleDirectoryReader(memoryTermConfig.path).load_data()