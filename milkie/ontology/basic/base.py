from abc import ABC, abstractmethod
from typing import Dict, Any

# 避免循环导入，使用类型提示字符串
# from .concept import Concept

class ConceptInstance(ABC):
    """概念实例的抽象基类 (如 Object, Relation)

    Attributes:
        concept (Concept): 此实例所属的概念定义。
        values (Dict[str, Any]): 实例成员及其对应的值。
    """

    def __init__(self, concept: 'Concept', values: Dict[str, Any]):
        """初始化概念实例

        Args:
            concept (Concept): 此实例所属的概念。
            values (Dict[str, Any]): 实例成员及其对应的值。

        Raises:
            TypeError: 如果 concept 不是 Concept 的实例。
            ValueError: 如果提供的 values 不符合 Concept 的成员或类型要求。
        """
        # 延迟导入以避免循环依赖
        from .concept import Concept

        if not isinstance(concept, Concept):
            raise TypeError("必须提供一个 Concept 实例")

        # 1. 验证成员是否存在且完整
        providedMembers = set(values.keys())
        if not concept.validateMembers(providedMembers):
            missing = set(concept.members.keys()) - providedMembers
            extra = providedMembers - set(concept.members.keys())
            errorMsg = f"实例的成员与 Concept '{concept.name}' 不匹配。"
            if missing:
                errorMsg += f" 缺少成员: {missing}."
            if extra:
                errorMsg += f" 多余成员: {extra}."
            raise ValueError(errorMsg)

        # 2. 验证成员值的类型
        for memberName, value in values.items():
            if not concept.validateMemberValue(memberName, value):
                expectedType = concept.members[memberName].name
                actualType = type(value).__name__
                raise TypeError(f"成员 '{memberName}' 的值类型错误。期望类型: {expectedType}, 实际类型: {actualType}")

        # 使用 object.__setattr__ 来避免触发自定义的 __setattr__
        object.__setattr__(self, 'concept', concept)
        object.__setattr__(self, 'values', values)

    def __getattr__(self, name: str) -> Any:
        """允许通过属性访问成员值"""
        if name == 'concept' or name == 'values':
             # 防止无限递归
            raise AttributeError
        
        try:
             # 尝试从 values 字典获取值
             return self.values[name]
        except KeyError:
             # 如果值不存在，抛出 AttributeError
             raise AttributeError(f"'{type(self).__name__}' 对象没有属性 '{name}' 或该属性未在 values 中定义")


    def __setattr__(self, name: str, value: Any) -> None:
        """允许通过属性设置成员值 (如果成员已定义在 Concept 中)"""
        if name in ['concept', 'values']:
            # 允许设置内部属性
            object.__setattr__(self, name, value)
        elif name in self.concept.members:
            # 验证新值的类型
            if not self.concept.validateMemberValue(name, value):
                expectedType = self.concept.members[name].name
                actualType = type(value).__name__
                raise TypeError(f"尝试设置的成员 '{name}' 的值类型错误。期望类型: {expectedType}, 实际类型: {actualType}")
            # 设置值
            self.values[name] = value
        else:
            raise AttributeError(f"无法设置属性 '{name}'，因为它不是 Concept '{self.concept.name}' 定义的成员")

    def __repr__(self) -> str:
        """返回实例的字符串表示"""
        return f"{type(self).__name__}(concept='{self.concept.name}', values={self.values})"

    # 可以添加其他通用的实例方法
    # 例如，获取特定成员的值、检查成员是否存在等

# 可以添加其他通用的实例方法
# 例如，获取特定成员的值、检查成员是否存在等 