from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

from milkie.config.config import IndexConfig, LongTermDocsetSource, DocsetConfig, DocsetTermConfig, DocsetType
from milkie.index.index import Index
from milkie.ontology.docset.docset import Docset
from milkie.settings import Settings

class DocsetWithIndex():

    def __init__(
            self,
            settings :Settings,
            docsetConfig :DocsetConfig,
            indexConfig :IndexConfig,
            serviceContext :ServiceContext):
        self.settings = settings
        self.docsetConfig = docsetConfig
        self.indexConfig = indexConfig
        self.serviceContext = serviceContext
        self.index = None
        self.docset = None

    def rebuildFromLocalDir(self, localDir :str):
        self.docsetConfig = DocsetConfig([
            DocsetTermConfig(
                type=DocsetType.LONG_TERM,
                source=LongTermDocsetSource.LOCAL,
                path=localDir)
        ])
        self.docset = None

    def getIndex(self):
        self._lazyBuildIndex()
        return self.index

    def getDocset(self):
        self._lazyBuildIndex()
        return self.docset

    def _lazyBuildIndex(self):
        if self.docset is None:
            self.docset = Docset(
                docsetTermConfigs=self.docsetConfig.docsetConfig, 
                serviceContext=self.serviceContext)

            denseIndex = VectorStoreIndex(
                self.docset.nodes,
                storage_context=self.docset.storageContext,
                service_context=self.serviceContext)
            
            self.index = Index(denseIndex)