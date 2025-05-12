from abc import ABC, abstractmethod
from typing import Any, Dict, List

# 修改导入，使用统一的 DataSourceType
from milkie.config.config import DataSourceType
# 导入 Concept 和 Mapping 以支持 scan 方法的类型提示
from milkie.ontology.mapping import Mapping


class DataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, name: str, type: DataSourceType, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.schema = None

    @abstractmethod
    def connect(self) -> Any:
        """建立到数据源的连接"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, List[str]]:
        """获取数据源的模式信息（例如，表和列）"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭数据源连接"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """测试数据源连接是否成功"""
        pass

    @abstractmethod
    def executeQuery(self, query: str, fetchColumns: bool = True) -> Dict[str, Any]:
        """执行query查询"""
        pass

    @abstractmethod
    def sampleData(self, conceptName: str, count: int = 10) -> List[Dict[str, Any]]:
        """根据概念获取样本数据"""
        pass

    @abstractmethod
    def scan(self) -> List[Mapping]:
        """扫描数据源 schema，生成 Concept 对象和对应的 Mapping

        返回一个 Mapping 列表，每个 Mapping 代表一个数据源实体（如表）
        到其对应 Concept 的映射关系。
        """
        pass

    @property
    @abstractmethod
    def type(self) -> DataSourceType:
        """返回数据源的类型 (使用 config.DataSourceType)"""
        pass

    # 可以根据需要添加更多通用方法，例如 execute_query 等 