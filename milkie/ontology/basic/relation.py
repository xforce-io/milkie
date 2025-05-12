from typing import Dict, Any

from .base import ConceptInstance
from .concept import Concept
# 假设 Relation 的成员值可能引用 Object
# from .object import Object

class Relation(ConceptInstance):
    """表示 Concept 的一个具体实例（关系）

    Relation 必须遵循其关联 Concept 定义的成员契约。
    成员的值通常用于连接 Object，但也可能是任何 Python 类型。
    """
    pass  # 所有基本功能都从 ConceptInstance 继承

    def __init__(self, concept: Concept, values: Dict[str, Any]):
        """初始化 Relation

        Args:
            concept (Concept): 此 Relation 所属的 Concept。
            values (Dict[str, Any]): Relation 成员及其对应的值。

        Raises:
            ValueError: 如果提供的 values 不符合 Concept 的成员要求。
            TypeError: 如果 concept 不是 Concept 的实例。
        """
        if not isinstance(concept, Concept):
            raise TypeError("必须提供一个 Concept 实例")

        provided_members = set(values.keys())
        if not concept.validateMembers(provided_members):
            missing = concept.members - provided_members
            extra = provided_members - concept.members
            error_msg = f"Relation 的成员与 Concept '{concept.name}' 不匹配。"
            if missing:
                error_msg += f" 缺少成员: {missing}."
            if extra:
                error_msg += f" 多余成员: {extra}."
            # 严格模式：不允许缺少或多余成员。
            raise ValueError(error_msg)

        self.concept = concept
        self.values = values

    def __getattr__(self, name: str) -> Any:
        """允许通过属性访问成员值"""
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"'{type(self).__name__}' 对象没有属性 '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """允许通过属性设置成员值 (如果成员已定义)"""
        if name in ['concept', 'values']:
            super().__setattr__(name, value)
        elif name in self.concept.members:
            self.values[name] = value
        else:
            raise AttributeError(f"无法设置属性 '{name}'，因为它不是 Concept '{self.concept.name}' 定义的成员")

    def __repr__(self) -> str:
        return f"Relation(concept='{self.concept.name}', values={self.values})" 