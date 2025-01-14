from typing import Optional, Set
from clients.bird.base_searcher import Node, NodeType, NodeExpansionRule
from clients.bird.base_sql_searcher import BaseSqlNodeType, BaseSqlSearcher
from clients.bird.config import Config
from clients.bird.logger import INFO, ERROR
from milkie.utils.data_utils import escape

class Text2SqlNodeType(BaseSqlNodeType):
    """Text2SQL特定的节点类型"""
    THOUGHT = "thought"

class Text2SqlSearcher(BaseSqlSearcher):
    def __init__(self, 
            query: str,
            max_iters: int,
            config: Optional[Config] = None):
        super().__init__(query, max_iters, config)
        
    def _create_root_node(self) -> Node:
        """创建根节点"""
        root = Node(type=Text2SqlNodeType.ROOT)
        root.data["query"] = self.query
        root.data["thought_count"] = 0
        return root
        
    def _init_expansion_rules(self):
        """初始化节点扩展规则"""
        # ROOT -> THOUGHT
        self.expansion_rules.append(NodeExpansionRule(
            source_type=Text2SqlNodeType.ROOT,
            target_type=Text2SqlNodeType.THOUGHT,
            min_expansions=1,
            max_expansions=self.config.search.max_thoughts
        ))
        
        # THOUGHT -> SQL
        self.expansion_rules.append(NodeExpansionRule(
            source_type=Text2SqlNodeType.THOUGHT,
            target_type=Text2SqlNodeType.SQL,
            min_expansions=self.config.search.min_sqls,
            max_expansions=self.config.search.max_sqls
        ))
        
    def _expand_node(self, node: Node, target_type: NodeType) -> Optional[Node]:
        """执行节点扩展"""
        if target_type == Text2SqlNodeType.THOUGHT:
            return self._expand_thought(node)
        elif target_type == Text2SqlNodeType.SQL:
            return self._expand_sql(node)
        return None
        
    def _expand_thought(self, node: Node) -> Node:
        """扩展THOUGHT节点"""
        node.data["thought_count"] += 1
        code = self._generate_thought_prompt(
            node.data["query"],
            node.get_error_patterns(),
            node.data["thought_count"]
        )
        thought = self._client.execute(code, self.config.agent.name)
        
        thought_node = Node(
            type=Text2SqlNodeType.THOUGHT,
            parent=node,
            depth=node.depth + 1
        )
        thought_node.data["query"] = node.data["query"]
        thought_node.data["thought"] = thought
        thought_node.data["sql_count"] = 0
        thought_node.data["success_count"] = 0
        
        INFO(f"Node[{node.id}] expanded to THOUGHT node[{thought_node.id}|{self._unnewline(thought)}]")
        return thought_node
       
    def _should_continue_expansion(self, node: Node, rule: NodeExpansionRule) -> bool:
        """判断是否应该继续扩展"""
        if node.type == Text2SqlNodeType.ROOT:
            return node.data["thought_count"] < rule.max_expansions
        elif node.type == Text2SqlNodeType.THOUGHT:
            return (node.data["sql_count"] < rule.max_expansions and 
                    (node.data["sql_count"] < rule.min_expansions or 
                     node.data["success_count"] == 0))
        return False
            
    def _check_node_completion(self, node: Node) -> bool:
        """检查节点是否完成"""
        if node.is_completed():
            return True
            
        if node.type == Text2SqlNodeType.ROOT:
            should_complete = node.data["thought_count"] >= self.config.search.max_thoughts
        elif node.type == Text2SqlNodeType.THOUGHT:
            should_complete = (node.data["sql_count"] >= self.config.search.max_sqls or
                             (node.data["sql_count"] >= self.config.search.min_sqls and
                              node.data["success_count"] > 0))
        else:
            should_complete = True
            
        if should_complete:
            node.mark_completed()
        return should_complete
        
    def _handle_expansion_error(self, node: Node, rule: NodeExpansionRule, error: Exception):
        """处理扩展错误"""
        error_msg = str(error)
        ERROR(f"Node[{node.id}] failed to expand to {rule.target_type}: {error_msg}")
        if node.type == Text2SqlNodeType.THOUGHT:
            node.data["error_patterns"].add(error_msg)

    def _generate_thought_prompt(self, query: str, error_patterns: Set[str], trial: int) -> str:
        """生成thought提示"""
        error_hints = ""
        if error_patterns:
            error_hints = "\n已有的错误模式：\n" + "\n".join(f"- {e}" for e in error_patterns)
            
        if self.config.search.table_desc_record_samples > 0:
            schema_desc_prompt = f'''可用schema解释 ```{self._db.descTablesFromQuery(query, self.config.search.table_desc_record_samples)}```'''
        else:
            schema_desc_prompt = ""
            
        return escape(f"""
    [{self.config.model.thought_model}] (trial: {trial}) 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决思路
    schema及问题 ```{query}```
    {schema_desc_prompt}
    已有错误模式 ```{error_hints}```

    请注意以下规则：
    1. 首先明确 query 中的问题问的 metric，不需要回答多余的信息
    2. 分析需要用到的表和字段，仅使用必要的表和字段,考虑表之间的关联关系
    3. 如果有错误模式，思考如何避免这些错误
    
    现在请输出你的分析和思考，请不要直接输出 sql：
""")