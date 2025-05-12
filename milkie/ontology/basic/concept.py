from typing import List, Set, Dict, Any, Type, Optional, Union
from enum import Enum, auto


class ConceptMemberType(Enum):
    """概念成员的类型"""
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    OBJECT = auto()  # 引用其他 Object
    RELATION = auto()  # 引用其他 Relation
    ANY = auto()  # 任意类型


class Concept:
    """定义概念的契约（结构）

    一个 Concept 定义了一组必须存在的成员名称及其类型。
    每个成员可以有类型约束，用于验证实例化时的值。
    """
    def __init__(self, name: str, members: Dict[str, ConceptMemberType]):
        """初始化 Concept

        Args:
            name (str): 概念的名称
            members (Dict[str, ConceptMemberType]): 成员名称及其类型的映射

        Raises:
            ValueError: 如果名称为空或没有定义成员
        """
        if not name:
            raise ValueError("概念名称不能为空")
        if not members:
            raise ValueError("概念必须至少定义一个成员")

        self.name = name
        self.members = members
        self.mappings = []

    def __repr__(self) -> str:
        members_str = {name: type.name for name, type in self.members.items()}
        return f"Concept(name='{self.name}', members={members_str})"

    def addMapping(self, mapping):
        self.mappings.append(mapping)

    def getDataSourceSchemas(self) -> Dict[str, Any]:
        """获取所有关联的数据源的 schema"""
        dataSourceSchemas = {}
        for mapping in self.mappings:
            dataSourceSchemas[mapping.space] = mapping.getDataSourceSchema()
        return dataSourceSchemas

    def validateMembers(self, provided_members: Set[str]) -> bool:
        """检查提供的成员集合是否完全匹配此 Concept 定义的要求

        Args:
            provided_members (Set[str]): 要验证的成员名称集合

        Returns:
            bool: 如果提供的成员与定义的成员完全匹配则返回 True
        """
        return set(self.members.keys()) == provided_members

    def validateMemberValue(self, memberName: str, value: Any) -> bool:
        """验证成员的值是否符合类型要求

        Args:
            member_name (str): 成员名称
            value (Any): 要验证的值

        Returns:
            bool: 如果值符合类型要求则返回 True

        Raises:
            KeyError: 如果成员名称不存在
        """
        if memberName not in self.members:
            raise KeyError(f"成员 '{memberName}' 未在概念 '{self.name}' 中定义")

        expectedType = self.members[memberName]
        
        # 如果类型是 ANY，直接返回 True
        if expectedType == ConceptMemberType.ANY:
            return True

        # 根据期望的类型进行验证
        if expectedType == ConceptMemberType.STRING:
            return isinstance(value, str)
        elif expectedType == ConceptMemberType.NUMBER:
            return isinstance(value, (int, float))
        elif expectedType == ConceptMemberType.BOOLEAN:
            return isinstance(value, bool)
        elif expectedType == ConceptMemberType.OBJECT:
            from .object import Object
            return isinstance(value, Object)
        elif expectedType == ConceptMemberType.RELATION:
            from .relation import Relation
            return isinstance(value, Relation)
        
        return False  # 未知类型

    def validateValues(self, values: Dict[str, Any]) -> bool:
        """验证所有值是否符合概念定义

        Args:
            values (Dict[str, Any]): 要验证的值字典

        Returns:
            bool: 如果所有值都符合要求则返回 True
        """
        if not self.validateMembers(set(values.keys())):
            return False
        
        return all(
            self.validateMemberValue(name, value)
            for name, value in values.items()
        ) 

    def toDict(self) -> Dict[str, Any]:
        """将概念转换为字典"""
        return {
            "name": self.name,
            "members": {name: type_.name for name, type_ in self.members.items()}
        }
