from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Set, Dict, Any
from collections import deque

from clients.bird.logger import INFO, ERROR

class NodeType(Enum):
    ROOT = "root"
    THOUGHT = "thought"
    SQL = "sql"

@dataclass
class ExpaFields:
    """用于扩展计算的字段"""
    query: Optional[str] = None
    thought: Optional[str] = None
    error_patterns: Set[str] = field(default_factory=set)

@dataclass
class StatisticsFields:
    """统计信息字段"""
    thought_count: int = 0  # THOUGHT 节点生成计数
    sql_count: int = 0      # SQL 节点生成计数
    success_count: int = 0   # 成功的 SQL 数量
    is_completed: bool = False  # 标记节点是否计算完成

@dataclass
class OtherFields:
    """其他字段"""
    sql: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    success: bool = True

@dataclass
class Node:
    type: NodeType
    parent: Optional['Node']
    children: List['Node']
    depth: int
    id: int = field(init=False)
    
    # 字段分组
    expa: ExpaFields = field(default_factory=ExpaFields)
    stats: StatisticsFields = field(default_factory=StatisticsFields)
    other: OtherFields = field(default_factory=OtherFields)
    
    def __post_init__(self):
        self.id = Node.next_id()
    
    @classmethod
    def reset_counter(cls):
        """重置节点 ID 计数器"""
        cls._id_counter = 0
        
    @classmethod
    def next_id(cls) -> int:
        if not hasattr(cls, '_id_counter'):
            cls._id_counter = 0
        cls._id_counter += 1
        return cls._id_counter
    
    def add_error_pattern(self, error: str):
        """添加错误模式并向上传播"""
        self.expa.error_patterns.add(error)
        if self.parent:
            self.parent.add_error_pattern(error)
            
    def mark_completed(self):
        """标记节点计算完成"""
        self.stats.is_completed = True
            
    def is_completed(self) -> bool:
        """检查节点是否已完成"""
        return self.stats.is_completed
        
    def increment_thought_count(self) -> int:
        """增加并返回 thought 计数"""
        self.stats.thought_count += 1
        return self.stats.thought_count
        
    def increment_sql_count(self) -> int:
        """增加并返回 sql 计数"""
        self.stats.sql_count += 1
        return self.stats.sql_count
        
    def increment_success_count(self) -> int:
        """增加并返回成功计数"""
        self.stats.success_count += 1
        return self.stats.success_count
        
    def should_continue_thought(self, max_thoughts: int) -> bool:
        """检查是否应该继续生成 THOUGHT"""
        return self.type == NodeType.ROOT and self.stats.thought_count < max_thoughts
        
    def should_continue_sql(self, min_sqls: int, max_sqls: int) -> bool:
        """检查是否应该继续生成 SQL"""
        if self.type != NodeType.THOUGHT or self.stats.sql_count >= max_sqls:
            return False
        return self.stats.sql_count < min_sqls or self.stats.success_count == 0
        
    def should_mark_completed(self, min_sqls: int, max_sqls: int, max_thoughts: int) -> bool:
        """检查是否应该标记为完成"""
        if self.type == NodeType.ROOT:
            return self.stats.thought_count >= max_thoughts
            
        if self.type == NodeType.THOUGHT:
            has_valid_result = any(
                child.type == NodeType.SQL and 
                child.other.success and 
                child.other.result != "[]" 
                for child in self.children
            )
            return (has_valid_result and self.stats.sql_count >= min_sqls) or \
                   self.stats.sql_count >= max_sqls or \
                   (self.stats.sql_count >= min_sqls and self.stats.success_count > 0)
                   
        return False

class SearchTree:
    def __init__(self, query: str, max_iters: int):
        Node.reset_counter()
        self.max_iters = max_iters
        self.iteration = 0
        
        # 创建根节点
        self.root = Node(
            type=NodeType.ROOT,
            parent=None,
            children=[],
            depth=0
        )
        self.root.expa.query = query
        INFO(f"Created ROOT node[{self.root.id}] with query: {query.replace(chr(10), '|')}")
        
        # 用深度优先搜索的栈
        self.stack = [self.root]
        # 叶子节点列表
        self.leaf_nodes = []
        
    def is_completed(self) -> bool:
        """检查是否所有 THOUGHT 节点都已完成或达到最大迭代次数"""
        if self.iteration >= self.max_iters:
            INFO(f"Reached maximum iterations: {self.max_iters}")
            return True
            
        # 获取所有 THOUGHT 节点
        thought_nodes = self.get_nodes_by_type(NodeType.THOUGHT)
        
        # 如果还没有 THOUGHT 节点，说明还没开始扩展，返回 False
        if not thought_nodes:
            return False
            
        # 检查所有 THOUGHT 节点是否都已完成
        for node in thought_nodes:
            if not node.stats.is_completed:
                return False
                
        return True
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """获取指定类型的所有节点"""
        result = []
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            if node.type == node_type:
                result.append(node)
            # 注意：为了保持遍历顺序，需要倒序添加子节点
            for child in reversed(node.children):
                stack.append(child)
            
        return result
    
    def get_best_sql(self) -> Optional[str]:
        """根据排序规则选择最佳的 SQL"""
        # 对叶子节点进行排序
        def node_priority(node: Node) -> tuple:
            has_result = node.type == NodeType.SQL and node.other.result is not None
            result_not_empty = has_result and node.other.result != "[]"
            return (has_result, result_not_empty, node.other.success)
        
        self.leaf_nodes.sort(key=node_priority, reverse=True)
        
        if not self.leaf_nodes:
            INFO("No leaf nodes found")
            return None
        
        # 获取所有有非空结果的节点
        valid_nodes = [
            node for node in self.leaf_nodes 
            if node.type == NodeType.SQL 
            and node.other.result is not None 
            and node.other.result != "[]"
        ]
        
        if not valid_nodes:
            # 如果没有非空结果，尝试找有空结果的成功节点
            empty_nodes = [
                node for node in self.leaf_nodes
                if node.type == NodeType.SQL
                and node.other.result is not None
                and node.other.result == "[]"
                and node.other.success
            ]
            
            if empty_nodes:
                selected_node = empty_nodes[0]
                INFO(f"No non-empty results found, selected SQL node[{selected_node.id}] with empty result")
                return selected_node.other.sql
                
            ERROR("No valid results found")
            return None
        
        # 统计结果出现次数
        result_counts = {}
        for node in valid_nodes:
            result_counts[node.other.result] = result_counts.get(node.other.result, 0) + 1
        
        # 找到出现次数最多的结果
        max_count = max(result_counts.values())
        most_common_results = [
            result for result, count in result_counts.items() 
            if count == max_count
        ]
        
        # 在出现次数最多的结果中，选择第一个应的 SQL
        for node in valid_nodes:
            if node.other.result in most_common_results:
                INFO(f"Selected SQL node[{node.id}] with most common result (count: {max_count})")
                return node.other.sql
        
        # 这种情况理论上不会发生
        ERROR("Failed to find SQL for most common result")
        return None 