from typing import List
from llama_index.core.service_context import ServiceContext
from llama_index.core.readers.file.base import SimpleDirectoryReader 
from llama_index.readers.file.unstructured.base import UnstructuredReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode

from milkie.config.config import LongTermDocsetSource, DocsetTermConfig, DocsetType


class Docset(object):
    def __init__(
            self, 
            docsetTermConfigs :List[DocsetTermConfig],
            serviceContext :ServiceContext):
        self.docSet = []
        for docsetTermConfig in docsetTermConfigs:
            if docsetTermConfig.source == LongTermDocsetSource.LOCAL:
                docs = self.__buildDocsFromLongTermLocal(docsetTermConfig)
                self.docSet.append(docs)
            else:
                raise Exception(f"Not supported long term docset type[{docsetTermConfig.source}]")

        self.serviceContext = serviceContext
        self.nodes = self.serviceContext.node_parser.get_nodes_from_documents(self.docSet[0])

        self.idToNodes = {}
        for node in self.nodes:
            node.text = node.text.strip().replace("\u3000", "").replace("\n\n", " ")
            self.idToNodes[node.node_id] = node
        
        self.storageContext = StorageContext.from_defaults()
        self.storageContext.docstore.add_documents(self.nodes)
    
    def getNodeFromId(self, id :str) -> BaseNode:
        return self.idToNodes.get(id)

    def getNextNode(self, node :BaseNode) -> BaseNode:
        return self.getNodeFromId(node.next_node.node_id) if node.next_node else None

    def getPrevNode(self, node :BaseNode) -> BaseNode:
        return self.getNodeFromId(node.prev_node.node_id) if node.prev_node else None
    
    def __buildDocsFromLongTermLocal(self, docsetTermConfig :DocsetTermConfig):
        loader = SimpleDirectoryReader(docsetTermConfig.path, file_extractor={
            ".txt" : UnstructuredReader()
        })
        return loader.load_data()

if __name__ == "__main__":
    docsetTermConfig = DocsetTermConfig(
        DocsetType.LONG_TERM,
        LongTermDocsetSource.LOCAL,
        "data/santi/",
    )

    docset = Docset([docsetTermConfig], ServiceContext.from_defaults())
    import pdb; pdb.set_trace()
    print(len(docset.nodes))