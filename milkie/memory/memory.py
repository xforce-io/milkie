from typing import List
from llama_index.legacy.service_context import ServiceContext
from llama_index.core import SimpleDirectoryReader 
from llama_index.readers.file.unstructured.base import UnstructuredReader
from llama_index.legacy.storage.storage_context import StorageContext
from llama_index.legacy.schema import BaseNode

from milkie.config.config import LongTermMemorySource, MemoryTermConfig, MemoryType


class Memory(object):
    def __init__(
            self, 
            memoryTermConfigs :List[MemoryTermConfig],
            serviceContext :ServiceContext):
        self.docSet = []
        for memoryTermConfig in memoryTermConfigs:
            if memoryTermConfig.source == LongTermMemorySource.LOCAL.name:
                docs = self.__buildDocsFromLongTermLocal(memoryTermConfig)
                self.docSet.append(docs)
            else:
                raise Exception(f"Not supported long term memory type[{memoryTermConfig.source}]")

        self.serviceContext = serviceContext
        self.nodes = self.serviceContext.node_parser.get_nodes_from_documents(self.docSet[0])

        self.idToNodes = {}
        for node in self.nodes:
            node.text = node.text.strip().replace("\u3000", "")
            self.idToNodes[node.node_id] = node
        
        self.storageContext = StorageContext.from_defaults()
        self.storageContext.docstore.add_documents(self.nodes)
    
    def getNodeFromId(self, id :str) -> BaseNode:
        return self.idToNodes.get(id)

    def getNextNode(self, node :BaseNode) -> BaseNode:
        return self.getNodeFromId(node.next_node.node_id) if node.next_node else None

    def getPrevNode(self, node :BaseNode) -> BaseNode:
        return self.getNodeFromId(node.prev_node.node_id) if node.prev_node else None
    
    def __buildDocsFromLongTermLocal(self, memoryTermConfig :MemoryTermConfig):
        loader = SimpleDirectoryReader(memoryTermConfig.path, file_extractor={
            ".txt" : UnstructuredReader()
        })
        return loader.load_data()

if __name__ == "__main__":
    memoryTermConfig = MemoryTermConfig(
        MemoryType.LONG_TERM.name,
        LongTermMemorySource.LOCAL.name,
        "data/santi/",
    )

    memory = Memory([memoryTermConfig], ServiceContext.from_defaults())
    import pdb; pdb.set_trace()
    print(len(memory.nodes))