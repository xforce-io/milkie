from typing import Dict, Any, Set, Optional, Iterator, Tuple
from abc import ABC

class Mapping:
    """定义数据源字段到 Concept 成员的映射关系
    
    此类维护数据源字段和概念成员之间的双向映射关系，
    并提供验证、查询和转换等功能。
    """

    def __init__(
            self, 
            dataSource: 'DataSource', 
            space: str,
            concept: 'Concept', 
            fieldToMemberMap: Dict[str, str]):
        """初始化 Mapping

        Args:
            dataSource (DataSource): 关联的数据源实例
            space (str): 关联的数据源空间
            concept (Concept): 关联的 Concept 实例
            fieldToMemberMap (Dict[str, str]): 从数据源字段名到 Concept 成员名的映射
                例如：{'column_name': 'memberName', 'another_col': 'otherMember'}

        Raises:
            TypeError: 如果参数类型不正确
            ValueError: 如果映射关系无效
        """
        # 导入放在这里，确保类型检查时可用，但避免顶层循环导入
        from .basic.concept import Concept
        from .datasource.datasource import DataSource

        if not isinstance(dataSource, DataSource):
            raise TypeError("dataSource 必须是 DataSource 的实例")
        if not isinstance(concept, Concept):
            raise TypeError("concept 必须是 Concept 的实例")
        if not isinstance(fieldToMemberMap, dict):
            raise TypeError("fieldToMemberMap 必须是一个字典")
        if not fieldToMemberMap:
            raise ValueError("fieldToMemberMap 不能为空")

        # 验证映射的成员是否存在于 Concept 中
        mappedMembers = set(fieldToMemberMap.values())
        conceptMembers = set(concept.members.keys())
        if not mappedMembers.issubset(conceptMembers):
            unknownMembers = mappedMembers - conceptMembers
            raise ValueError(f"映射包含未在 Concept '{concept.name}' 中定义的成员: {unknownMembers}")
        
        # 验证字段名的唯一性
        if len(set(fieldToMemberMap.keys())) != len(fieldToMemberMap):
            raise ValueError("字段名映射存在重复")

        # 验证成员名的唯一性（可选，取决于是否允许多个字段映射到同一个成员）
        if len(set(fieldToMemberMap.values())) != len(fieldToMemberMap):
            raise ValueError("成员名映射存在重复，每个字段必须映射到唯一的成员")

        self.dataSource = dataSource
        self.space = space
        self.concept = concept
        concept.addMapping(self)

        self.fieldToMemberMap = fieldToMemberMap

        self.dataSourceSchema = None
        
        # 构建反向映射，用于快速查找
        self._memberToFieldMap = {member: field for field, member in fieldToMemberMap.items()}

    def __repr__(self) -> str:
        return (f"Mapping(dataSource='{self.dataSource.name}', "
                f"space='{self.space}', "
                f"concept='{self.concept.name}', "
                f"map={self.fieldToMemberMap})")

    def getMemberForField(self, fieldName: str) -> Optional[str]:
        """根据数据源字段名获取对应的 Concept 成员名

        Args:
            fieldName (str): 数据源字段名

        Returns:
            Optional[str]: 对应的成员名，如果不存在则返回 None
        """
        return self.fieldToMemberMap.get(fieldName)

    def getFieldForMember(self, memberName: str) -> Optional[str]:
        """根据 Concept 成员名获取对应的数据源字段名

        Args:
            memberName (str): 概念成员名

        Returns:
            Optional[str]: 对应的字段名，如果不存在则返回 None
        """
        return self._memberToFieldMap.get(memberName)

    def getAllMappings(self) -> Iterator[Tuple[str, str]]:
        """获取所有字段到成员的映射关系

        Returns:
            Iterator[Tuple[str, str]]: (field_name, member_name) 元组的迭代器
        """
        return iter(self.fieldToMemberMap.items())

    def getUnmappedMembers(self) -> Set[str]:
        """获取未映射的概念成员

        Returns:
            Set[str]: 未被映射的概念成员名集合
        """
        return set(self.concept.members.keys()) - set(self.fieldToMemberMap.values())

    def getDataSourceSchema(self) -> Dict[str, Any]:
        """获取数据源的 schema"""
        if self.dataSourceSchema:
            return self.dataSourceSchema

        schema = {}
        dataSourceSchema = self.dataSource.get_schema()
        for field, member in self.fieldToMemberMap.items():
            schemaPerSpace = dataSourceSchema[self.space]
            for singleSchema in schemaPerSpace:
                if singleSchema['name'] == field:
                    schema[field] = singleSchema['type']
                    break
        self.dataSourceSchema = schema
        return schema

    def validateFieldValue(self, fieldName: str, value: Any) -> bool:
        """验证字段值是否符合对应成员的类型要求

        Args:
            fieldName (str): 数据源字段名
            value (Any): 要验证的值

        Returns:
            bool: 如果值符合类型要求则返回 True

        Raises:
            KeyError: 如果字段名不存在
        """
        memberName = self.getMemberForField(fieldName)
        if memberName is None:
            raise KeyError(f"字段 '{fieldName}' 未在映射中定义")
        return self.concept.validateMemberValue(memberName, value)

    def transformFieldValues(self, fieldValues: Dict[str, Any]) -> Dict[str, Any]:
        """将数据源字段值转换为概念成员值

        Args:
            fieldValues (Dict[str, Any]): 字段名到值的映射

        Returns:
            Dict[str, Any]: 成员名到值的映射

        Raises:
            ValueError: 如果存在无效的字段值
        """
        memberValues = {}
        for fieldName, value in fieldValues.items():
            memberName = self.getMemberForField(fieldName)
            if memberName is None:
                continue  # 跳过未映射的字段
            if not self.validateFieldValue(fieldName, value):
                raise ValueError(f"字段 '{fieldName}' 的值类型不符合成员 '{memberName}' 的要求")
            memberValues[memberName] = value
        return memberValues 