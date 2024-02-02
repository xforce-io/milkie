from typing import List
from llama_index import ServiceContext, SimpleDirectoryReader, StorageContext
from milkie.config.config import LongTermMemorySource, MemoryTermConfig


class Memory(object):
    def __init__(
            self, 
            memoryTermConfigs :List[MemoryTermConfig],
            serviceContext :ServiceContext):
        import pdb
        pdb.set_trace()
        
        self.docSet = []
        for memoryTermConfig in memoryTermConfigs:
            if memoryTermConfig.source == LongTermMemorySource.LOCAL.name:
                docs = self.__buildDocsFromLongTermLocal(memoryTermConfig)
                self.docSet.append(docs)
            else:
                raise Exception(f"Not supported long term memory type[{memoryTermConfig.source}]")

        self.serviceContext = serviceContext
        self.nodes = self.serviceContext.node_parser.get_nodes_from_documents(self.docSet[0])
        self.storageContext = StorageContext.from_defaults()
        self.storageContext.docstore.add_documents(self.nodes)
    
    def __buildDocsFromLongTermLocal(self, memoryTermConfig :MemoryTermConfig):
        return SimpleDirectoryReader(memoryTermConfig.path).load_data()