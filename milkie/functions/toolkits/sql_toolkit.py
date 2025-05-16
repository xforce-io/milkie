from typing import List
from milkie.functions.openai_function import OpenAIFunction
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext


class SQLToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)

    def getName(self) -> str:
        return "SQLToolkit"

    def executeSQL(self, datasource: str, sql: str) -> str:
        """
        在指定的数据源中执行SQL语句，并返回结果。
        调用格式为
        ```
            datasource: 数据源名称
            sql: SQL语句
        ```

        Args:
            datasource (str): 数据源名称
            sql (str): SQL语句
        """
        theDataSource = self.globalContext.getOntology().getDataSource(datasource)
        if theDataSource is None:
            return f"数据源 {datasource} 不存在"

        try:
            return theDataSource.executeQuery(sql)
        except Exception as e:
            return f"执行SQL语句失败: {e}"

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.executeSQL)
        ]