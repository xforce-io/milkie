from typing import List, Optional, Any
from abc import ABC, abstractmethod
import uuid

class NodeType(object):
    """基础节点类型，可被继承扩展"""
    BASE = "base"

class Node:
    def __init__(self, 
            type: str,
            parent: Optional['Node'] = None,
            children: Optional[List['Node']] = None,
            depth: int = 0):
        self.id = str(uuid.uuid4())
        self.type = type
        self.parent = parent
        self.children = children or []
        self.depth = depth
        self.completed = False
        self.high_confidence = False
        self.successful_children = 0
        
        # 扩展数据，由具体实现定义
        self.data = {}
        # 错误模式集合
        self.error_patterns = set()
        
    def get_num_children(self) -> int:
        """获取子节点数量"""
        return len(self.children)
        
    def is_completed(self) -> bool:
        return self.completed
        
    def mark_completed(self):
        self.completed = True
        
    def add_successful_child(self):
        self.successful_children += 1
        
    def get_successful_children(self) -> int:
        return self.successful_children
        
    def add_error_pattern(self, error: str):
        """添加错误模式并传播到父节点"""
        self.error_patterns.add(error)
        if self.parent:
            self.parent.add_error_pattern(error)
        
    def get_error_patterns(self) -> set:
        """获取当前节点的错误模式"""
        return self.error_patterns
        
    def get_parent_data(self, key: str, default: Any = None) -> Any:
        """从父节点获取数据
        
        Args:
            key: 数据键名
            default: 如果键不存在时的默认值
            
        Returns:
            Any: 父节点中对应键的数据，如果父节点不存在或键不存在则返回默认值
        """
        if self.parent and key in self.parent.data:
            return self.parent.data[key]
        return default
        
    def get_ancestor_data(self, key: str, default: Any = None) -> Any:
        """从祖先节点获取数据（向上查找直到找到第一个包含该键的节点）
        
        Args:
            key: 数据键名
            default: 如果键不存在时的默认值
            
        Returns:
            Any: 最近的祖先节点中对应键的数据，如果所有祖先都不存在该键则返回默认值
        """
        current = self
        while current.parent:
            current = current.parent
            if key in current.data:
                return current.data[key]
        return default

class NodeExpansionRule:
    """定义节点扩展规则"""
    def __init__(self, 
            source_type: NodeType,
            target_type: NodeType,
            min_expansions: int = 1,
            max_expansions: int = 1):
        self.source_type = source_type
        self.target_type = target_type
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions

class BaseSearchTree(ABC):
    def __init__(self, max_iters: int):
        self.max_iters = max_iters
        self.iteration = 0
        self.root = self._create_root_node()
        self.stack: List[Node] = []
        self.leaf_nodes: List[Node] = []
        
        # 节点扩展规则
        self.expansion_rules: List[NodeExpansionRule] = []
        self._init_expansion_rules()
    
    @abstractmethod
    def _create_root_node(self) -> Node:
        """创建根节点"""
        pass
        
    @abstractmethod
    def _init_expansion_rules(self):
        """初始化节点扩展规则"""
        pass
        
    @abstractmethod
    def _expand_node(self, node: Node, target_type: NodeType) -> Optional[Node]:
        """执行具体的节点扩展逻辑"""
        pass
        
    def _should_continue_expansion(self, node: Node, rule: NodeExpansionRule) -> bool:
        """判断是否应该继续扩展"""
        return (node.get_num_children() < rule.max_expansions and 
                (node.get_num_children() < rule.min_expansions or 
                    node.get_successful_children() == 0))
        
    @abstractmethod
    def _process_expansion_result(self, source_node: Node, new_node: Node):
        """处理扩展结果"""
        pass
        
    def _handle_expansion_error(self, node: Node, rule: NodeExpansionRule, error: Exception):
        """处理节点扩展错误，提供默认实现"""
        error_msg = str(error)
        node.add_error_pattern(error_msg)

    def forward_step(self):
        """执行一次前向扩展，采用广度优先策略"""
        current_level = []
        next_level = []
        
        # 初始状态或继续处理当前层级
        if not self.stack:
            if not self.root.is_completed():
                current_level.append(self.root)
        else:
            current_level = self.stack[:]
            
        self.stack = []  # 清空栈，准备收集下一层级的节点
        
        # 处理当前层级的所有节点
        for node in current_level:
            if node.is_completed():
                continue
                
            # 查找适用的扩展规则
            for rule in self.expansion_rules:
                if rule.source_type == node.type :
                    if self._should_continue_expansion(node, rule):
                        try:
                            new_node = self._expand_node(node, rule.target_type)
                            if new_node:
                                node.children.append(new_node)
                                self._process_expansion_result(node, new_node)
                            
                            # 处理高置信度节点
                            if new_node.high_confidence:
                                new_node.mark_completed()
                                if new_node not in self.leaf_nodes:
                                    self.leaf_nodes.append(new_node)
                            else:
                                next_level.append(new_node)
                        except Exception as e:
                            self._handle_expansion_error(node, rule, e)
                    else:
                        node.mark_completed()
                        
        # 检查节点是否需要继续探索
        for node in current_level:
            if not self._check_node_completion(node):
                next_level.append(node)
                
        # 更新下一轮要处理的节点
        self.stack = next_level
        
    def is_completed(self) -> bool:
        """判断搜索是否完成"""
        return (self.iteration >= self.max_iters or 
                (not self.stack and self.root.is_completed()))
                
    def _check_node_completion(self, node: Node) -> bool:
        """检查节点是否应该标记为完成"""
        return node.is_completed()
