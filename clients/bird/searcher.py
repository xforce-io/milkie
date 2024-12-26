from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from collections import deque

from milkie.sdk.agent_client import AgentClient
from milkie.sdk.config_server import ConfigServer
from clients.bird.config import Config
from clients.bird.database import Database
from clients.bird.logger import logger

class NodeType(Enum):
    ROOT = "root"
    THOUGHT = "thought"
    SQL = "sql"
    RESULT = "result"

@dataclass
class Node:
    type: NodeType
    parent: Optional['Node']
    children: List['Node']
    depth: int
    success: bool = True
    
    # 不同类型节点的特定属性
    query: Optional[str] = None  # ROOT
    thought: Optional[str] = None  # THOUGHT
    sql: Optional[str] = None  # SQL
    result: Optional[str] = None  # RESULT

class Searcher:
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config
        self._db = Database(config.database)
        self._client = AgentClient(ConfigServer(config.agent.addr))
        self.agent_name = config.agent.name
        
    def thought(self, query: str) -> str:
        return f"""
Given the query: {query}
Think about how to solve this query step by step.
Output your thought process in a clear and structured way.
"""

    def sql(self, query: str, thought: str) -> str:
        return f"""
Given the query: {query}
And the thought process: {thought}
Generate a SQL query that can help answer this question.
The SQL should be executable and return meaningful results.
"""

    def inference(self, query: str) -> str:
        try:
            # 创建根节点
            root = Node(
                type=NodeType.ROOT,
                parent=None,
                children=[],
                depth=0,
                query=query
            )
            
            # 使用队列进行广度优先搜索
            queue = deque([root])
            leaf_nodes = []
            
            while queue:
                node = queue.popleft()
                
                # 根据节点类型进行扩展
                if node.type == NodeType.ROOT:
                    # 扩展为 Thought 节点
                    for _ in range(self.config.search.max_thoughts):
                        try:
                            code = self.thought(node.query)
                            thought = self._client.execute(code, self.agent_name)
                            
                            thought_node = Node(
                                type=NodeType.THOUGHT,
                                parent=node,
                                children=[],
                                depth=node.depth + 1,
                                thought=thought
                            )
                            node.children.append(thought_node)
                            queue.append(thought_node)
                        except Exception as e:
                            logger.error(f"Error generating thought: {str(e)}")
                            # 如果生成失败，创建一个失败的节点
                            failed_node = Node(
                                type=NodeType.THOUGHT,
                                parent=node,
                                children=[],
                                depth=node.depth + 1,
                                success=False,
                                thought=str(e)
                            )
                            node.children.append(failed_node)
                            leaf_nodes.append(failed_node)
                            
                elif node.type == NodeType.THOUGHT:
                    # 扩展为 SQL 节点
                    for _ in range(self.config.search.max_sqls):
                        try:
                            code = self.sql(node.parent.query, node.thought)
                            sql = self._client.execute(code, self.agent_name)
                            
                            sql_node = Node(
                                type=NodeType.SQL,
                                parent=node,
                                children=[],
                                depth=node.depth + 1,
                                sql=sql
                            )
                            node.children.append(sql_node)
                            queue.append(sql_node)
                        except Exception as e:
                            logger.error(f"Error generating SQL: {str(e)}")
                            failed_node = Node(
                                type=NodeType.SQL,
                                parent=node,
                                children=[],
                                depth=node.depth + 1,
                                success=False,
                                sql=str(e)
                            )
                            node.children.append(failed_node)
                            leaf_nodes.append(failed_node)
                            
                elif node.type == NodeType.SQL:
                    # 扩展为 Result 节点
                    result = self._db.execsql(node.sql)
                    if result is None:
                        # SQL执行错误
                        failed_node = Node(
                            type=NodeType.RESULT,
                            parent=node,
                            children=[],
                            depth=node.depth + 1,
                            success=False,
                            result="SQL execution failed"
                        )
                        node.children.append(failed_node)
                        leaf_nodes.append(failed_node)
                    else:
                        result_node = Node(
                            type=NodeType.RESULT,
                            parent=node,
                            children=[],
                            depth=node.depth + 1,
                            result=result
                        )
                        node.children.append(result_node)
                        leaf_nodes.append(result_node)
                
                elif node.type == NodeType.RESULT:
                    # Result 节点是叶子节点，不需要扩展
                    if node not in leaf_nodes:
                        leaf_nodes.append(node)
            
            # 对叶子节点进行排序
            def node_priority(node: Node) -> tuple:
                has_result = node.type == NodeType.RESULT
                result_not_empty = has_result and node.result and node.result != "[]"
                return (has_result, result_not_empty, node.success)
            
            leaf_nodes.sort(key=node_priority, reverse=True)
            
            # 找到优先级最高的叶子节点
            if not leaf_nodes:
                return None
                
            best_node = leaf_nodes[0]
            
            # 回溯找到对应的 SQL 节点
            current = best_node
            while current and current.type != NodeType.SQL:
                current = current.parent
                
            return current.sql if current else None
            
        except Exception as e:
            logger.error(f"Error in inference: {str(e)}")
            raise
