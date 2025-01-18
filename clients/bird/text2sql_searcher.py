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
        return root
        
    def _init_expansion_rules(self):
        """初始化节点扩展规则"""
        # ROOT -> THOUGHT
        self.expansion_rules.append(NodeExpansionRule(
            source_type=Text2SqlNodeType.ROOT,
            target_type=Text2SqlNodeType.THOUGHT,
            min_expansions=1,
            max_expansions=self.config.search.text2sql.max_thoughts
        ))
        
        # THOUGHT -> SQL
        self.expansion_rules.append(NodeExpansionRule(
            source_type=Text2SqlNodeType.THOUGHT,
            target_type=Text2SqlNodeType.SQL,
            min_expansions=self.config.search.text2sql.min_sqls,
            max_expansions=self.config.search.text2sql.max_sqls
        ))
        
    def _expand_node(self, node: Node, target_type: NodeType) -> Optional[Node]:
        """执行节点扩展"""
        if target_type == Text2SqlNodeType.THOUGHT:
            return self._expand_thought(node)
        elif target_type == Text2SqlNodeType.SQL:
            return self._expand_sql(node, node.data["thought"])
        return None
        
    def _expand_thought(self, node: Node) -> Node:
        """扩展THOUGHT节点"""
        code = self._generate_thought_prompt(
            node.data["query"],
            node.get_error_patterns(),
            node.get_num_children()
        )
        thought = self._client.execute(code, self.config.agent.name)
        
        thought_node = Node(
            type=Text2SqlNodeType.THOUGHT,
            parent=node,
            depth=node.depth + 1
        )
        thought_node.data["query"] = node.data["query"]
        thought_node.data["thought"] = thought

        INFO(f"Node[{node.id}] expanded to THOUGHT node[{thought_node.id}|{self._unnewline(thought)}]")
        return thought_node
       
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
            error_hints = "已有的错误模式 ```" + "/".join(f"- {e}" for e in error_patterns) + "```"
            
        if self.config.search.table_desc_record_samples > 0:
            schema_desc_prompt = f'''可用schema解释 ```{self._db.descTablesFromQuery(query, self.config.search.table_desc_record_samples)}```'''
        else:
            schema_desc_prompt = ""
            
        return escape(f"""
    [{self.config.model.thought_model}] (trial: {trial}) 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决思路
    schema及问题 ```{query}```
    {error_hints}
    {schema_desc_prompt}

    请注意以下规则：
    1. 首先明确 query 中的问题问的 metric，不需要回答多余的信息
    2. 分析需要用到的表和字段，仅使用必要的表和字段,考虑表之间的关联关系
    3. 如果有错误模式，思考如何避免这些错误
    
    现在请输出你的分析和思考，请不要直接输出 sql：
""")