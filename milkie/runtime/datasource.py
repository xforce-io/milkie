from milkie.config.config import GlobalConfig
from milkie.retrieval.retrieval import RetrievalModule


class DataSource:
    def __init__(self, globalConfig: GlobalConfig):
        self.globalConfig = globalConfig
        self.mainRetriever :RetrievalModule = None

    def getMainRetriever(self):
        return self.mainRetriever

    def setMainRetriever(self, mainRetriever: RetrievalModule):
        self.mainRetriever = mainRetriever
