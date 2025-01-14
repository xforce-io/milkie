from typing import Optional
from clients.bird.base_searcher import Node, NodeExpansionRule
from clients.bird.config import Config
from clients.bird.base_sql_searcher import BaseSqlSearcher, BaseSqlNodeType
from milkie.utils.data_utils import escape

class TaskAlignmentNodeType(BaseSqlNodeType):
    """任务对齐特定的节点类型"""
    DUMMY_SQL = "dummy_sql"
    SCHEMA_LINKING = "schema_linking"
    SYMBOLIC_REPR = "symbolic_repr"

class TaskAlignmentSearcher(BaseSqlSearcher):
    def __init__(self, 
            query: str,
            max_iters: int,
            config: Optional[Config] = None):
        super().__init__(query, max_iters, config)

    def _create_root_node(self) -> Node:
        """创建根节点"""
        root = Node(type=TaskAlignmentNodeType.ROOT)
        root.data["query"] = self.query
        return root
    
    def _init_expansion_rules(self):
        """初始化节点扩展规则"""
        # ROOT -> DUMMY_SQL
        self.expansion_rules.append(NodeExpansionRule(
            source_type=TaskAlignmentNodeType.ROOT,
            target_type=TaskAlignmentNodeType.DUMMY_SQL,
            min_expansions=1,
            max_expansions=self.config.search.task_alignment.max_dummy_sqls
        ))
        
        # DUMMY_SQL -> SCHEMA_LINKING
        self.expansion_rules.append(NodeExpansionRule(
            source_type=TaskAlignmentNodeType.DUMMY_SQL,
            target_type=TaskAlignmentNodeType.SCHEMA_LINKING,
            min_expansions=1,
            max_expansions=1
        ))
        
        # SCHEMA_LINKING -> SYMBOLIC_REPR
        self.expansion_rules.append(NodeExpansionRule(
            source_type=TaskAlignmentNodeType.SCHEMA_LINKING,
            target_type=TaskAlignmentNodeType.SYMBOLIC_REPR,
            min_expansions=1,
            max_expansions=self.config.search.task_alignment.max_symbolic_reprs
        ))
        
        # SYMBOLIC_REPR -> SQL
        self.expansion_rules.append(NodeExpansionRule(
            source_type=TaskAlignmentNodeType.SYMBOLIC_REPR,
            target_type=TaskAlignmentNodeType.SQL,
            min_expansions=1,
            max_expansions=self.config.search.task_alignment.max_sqls
        ))

    def _expand_dummy_sql(self, node: Node) -> Node:
        """扩展dummy_sql节点"""
        node.data["dummy_sql_count"] += 1
        code = self._generate_dummy_sql_prompt(node.data["query"])
        dummy_sql = self._client.execute(code, self.config.agent.name)
        return Node(
            type=TaskAlignmentNodeType.DUMMY_SQL,
            parent=node,
            depth=node.depth + 1,
            data={"dummy_sql": dummy_sql}
        )
        
    def _expand_schema_linking(self, node: Node) -> Node:
        """扩展schema_linking节点"""
        # 从父节点获取 SQL
        sql = node.data["dummy_sql"]
        # 解析 SQL 中的 table.field 信息
        schema_linking = self._parse_schema_linking(sql)
        return Node(
            type=TaskAlignmentNodeType.SCHEMA_LINKING,
            parent=node,
            depth=node.depth + 1,
            data={"schema_linking": schema_linking}
        )
        
    def _expand_symbolic_repr(self, node: Node) -> Node:
        """扩展symbolic_repr节点"""
        # 从父节点获取 schema_linking
        schema_linking = node.data["schema_linking"]
        # 生成符号化表示
        symbolic_repr = self._generate_symbolic_repr_prompt(
            self.query,
            schema_linking
        )
        return Node(
            type=TaskAlignmentNodeType.SYMBOLIC_REPR,
            parent=node,
            depth=node.depth + 1,
            data={"symbolic_repr": symbolic_repr}
        )

    def _expand_sql(self, node: Node) -> Node:
        """扩展sql节点"""
        # 从父节点获取 symbolic_repr
        symbolic_repr = node.data["symbolic_repr"]
        code = self._generate_sql_prompt(
            self.query,
            symbolic_repr
        )
        sql = self._client.execute(code, self.config.agent.name)
        return Node(
            type=TaskAlignmentNodeType.SQL,
            parent=node,
            depth=node.depth + 1,
            data={"sql": sql}
        )
        
    def _parse_schema_linking(self, sql: str) -> list[str]:
        """解析SQL中的表和字段关系
        
        Args:
            sql: SQL语句
            
        Returns:
            list[str]: 表和字段关系列表，格式为 ["table.field", ...]
            
        Raises:
            ValueError: 当没有解析到任何schema时抛出
        """
        import sqlparse
        from sqlparse.sql import Identifier
        
        schema_links = []
        
        # 解析SQL
        parsed = sqlparse.parse(sql)[0]
        
        def process_token_list(tokens):
            for token in tokens:
                if isinstance(token, Identifier):
                    parts = [t.value for t in token.tokens if not t.is_whitespace]
                    if len(parts) == 3 and parts[1] == '.':  # table.field 形式
                        table, _, field = parts
                        # 移除可能的引号
                        table = table.strip('`"\'')
                        field = field.strip('`"\'')
                        schema_links.append(f"{table}.{field}")
                elif hasattr(token, 'tokens'):
                    process_token_list(token.tokens)
                    
        # 处理所有token
        process_token_list(parsed.tokens)
        
        if not schema_links:
            raise ValueError("No schema linking found in SQL")
            
        return schema_links

    def _generate_dummy_sql_prompt(self, query: str) -> str:
        """生成dummy_sql提示"""
        return escape(f"""
        [{self.config.model.sql_model}] 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决SQL
        schema及问题 ```{query}```

        结果 SQL 如下：
        """)

    def _generate_symbolic_repr_prompt(
            self, 
            query: str, 
            schema_linking: list[str]) -> str:
        """生成symbolic_repr提示"""
        return escape(f"""
        [{self.config.model.sql_model}] 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决pandas代码
        schema及问题 ```{query}```
        schema_linking ```{" ".join(schema_linking)}```
        结果 pandas 代码如下：
        """)
    
    def _generate_sql_prompt(self, query: str, symbolic_repr: str) -> str:
        """生成sql提示"""
        return escape(f"""
        [{self.config.model.sql_model}] 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决SQL
        schema及问题 ```{query}```
        pandas 代码 ```{symbolic_repr}```
        结果 SQL 如下：
        """)
