from typing import Dict, Any

from .base import ConceptInstance
from .concept import Concept


class Object(ConceptInstance):
    """表示 Concept 的一个具体实例（对象）

    Object 必须遵循其关联 Concept 定义的成员契约。
    成员的值可以是任何 Python 类型，或者引用其他 Object/Relation。
    """
    pass  # 所有基本功能都从 ConceptInstance 继承 