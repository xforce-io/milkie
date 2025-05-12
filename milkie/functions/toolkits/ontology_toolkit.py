from typing import List
from milkie.functions.openai_function import OpenAIFunction
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext

class OntologyToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

    def getName(self) -> str:
        return "OntologyToolkit"

    def getDesc(self) -> str:
        return "OntologyToolkit"

    def getAllConcepts(self) -> str:
        """
        获取本体模型中所有概念的描述

        Args:
            None

        Returns:
            str: 本体模型中所有概念的描述
        """
        return self.globalContext.getOntology().getAllConceptsDescription()

    def getSampleData(self, conceptNames: List[str]) -> str:
        """
        获取本体模型中指定概念的样本数据

        Args:
            conceptNames (List[str]): 概念名称列表

        Returns:
            str: 本体模型中指定概念的样本数据
        """
        return self.globalContext.getOntology().sampleData(conceptNames)

    def getDataSourceSchemas(self, conceptNames: List[str]) -> str:
        """
        获取本体模型中指定概念的数据源的 schema

        Args:
            conceptNames (List[str]): 概念名称列表
        """
        dataSourceSchemas = []
        for conceptName in conceptNames:
            concept = self.globalContext.getOntology().getConcept(conceptName)
            if not concept:
                continue
            dataSourceSchemas.append(concept.getDataSourceSchemas())
        return dataSourceSchemas

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.getAllConcepts),
            OpenAIFunction(self.getSampleData),
            OpenAIFunction(self.getDataSourceSchemas)
        ]