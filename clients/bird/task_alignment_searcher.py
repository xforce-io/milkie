from typing import Optional, List, Dict, Tuple
from clients.bird.base_searcher import Node, NodeExpansionRule, NodeType
from clients.bird.config import Config
from clients.bird.base_sql_searcher import BaseSqlSearcher, BaseSqlNodeType
from clients.bird.logger import INFO
from milkie.utils.data_utils import escape
import logging
import sqlparse
from sqlparse.sql import Identifier, Token, TokenList
from sqlparse.tokens import Keyword, Name, Punctuation

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

    def _expand_node(self, node: Node, target_type: NodeType) -> Optional[Node]:
        """执行节点扩展"""
        if target_type == TaskAlignmentNodeType.DUMMY_SQL:
            return self._expand_dummy_sql(node)
        elif target_type == TaskAlignmentNodeType.SCHEMA_LINKING:
            return self._expand_schema_linking(node)
        elif target_type == TaskAlignmentNodeType.SYMBOLIC_REPR:
            return self._expand_symbolic_repr(node)
        elif target_type == TaskAlignmentNodeType.SQL:
            return self._expand_sql(node, node.data["symbolic_repr"])
        return None

    def _expand_dummy_sql(self, node: Node) -> Node:
        """扩展dummy_sql节点"""
        code = self._generate_dummy_sql_prompt(node.data["query"])
        dummy_sql = self._client.execute(code, self.config.agent.name)
        node.add_successful_child()
        newNode = Node(
            type=TaskAlignmentNodeType.DUMMY_SQL,
            parent=node,
            depth=node.depth + 1,
            data={"dummy_sql": dummy_sql}
        )
        INFO(f"Node[{node.id}] expanded to dummy_sql node[{newNode.id}|{dummy_sql}]")
        return newNode
        
    def _expand_schema_linking(self, node: Node) -> Node:
        """扩展schema_linking节点
        
        从SQL中提取schema links并验证它们在数据库中是否存在。
        如果数据库连接失败，则跳过验证直接返回所有schema links。
        """
        # 从父节点获取 SQL
        sql = node.data["dummy_sql"]
        # 解析 SQL 中的 table.field 信息
        schema_linking = self._parse_schema_linking(sql)
        
        try:
            # 验证并过滤 schema links
            valid_schema_links = []
            for link in schema_linking:
                # 分割表名和字段名
                table_field = link.split('.')
                if len(table_field) != 2:
                    continue
                    
                table, field = table_field
                # 去除反引号
                table = table.strip('`')
                field = field.strip('`')
                
                try:
                    # 检查表是否存在
                    if not self._db.table_exists(table):
                        INFO(f"表 {table} 不存在")
                        continue
                        
                    # 检查字段是否存在
                    if not self._db.field_exists(table, field):
                        INFO(f"字段 {field} 在表 {table} 中不存在")
                        continue
                        
                    # 表和字段都存在，保留这个 schema link
                    valid_schema_links.append(link)
                    
                except Exception as e:
                    INFO(f"验证 schema link 时发生错误: {link}, 错误信息: {str(e)}")
                    continue
                    
            if not valid_schema_links:
                # 如果没有有效的 schema links，使用所有解析出的 links
                INFO("没有找到有效的 schema links，使用所有解析出的 links")
                valid_schema_links = schema_linking
                
        except Exception as e:
            # 如果数据库验证过程出错，使用所有解析出的 links
            INFO(f"验证 schema links 时发生错误: {str(e)}，使用所有解析出的 links")
            valid_schema_links = schema_linking
            
        node.add_successful_child()
        newNode = Node(
            type=TaskAlignmentNodeType.SCHEMA_LINKING,
            parent=node,
            depth=node.depth + 1,
            data={"schema_linking": valid_schema_links}
        )
        INFO(f"Node[{node.id}] expanded to schema_linking node[{newNode.id}|{valid_schema_links}]")
        return newNode
        
    def _expand_symbolic_repr(self, node: Node) -> Node:
        """扩展symbolic_repr节点"""
        # 从父节点获取 schema_linking
        schema_linking = node.data["schema_linking"]
        # 生成符号化表示
        symbolic_repr_prompt = self._generate_symbolic_repr_prompt(
            self.query,
            schema_linking
        )
        symbolic_repr = self._client.execute(symbolic_repr_prompt, self.config.agent.name)
        node.add_successful_child()
        newNode = Node(
            type=TaskAlignmentNodeType.SYMBOLIC_REPR,
            parent=node,
            depth=node.depth + 1,
            data={"symbolic_repr": symbolic_repr}
        )
        INFO(f"Node[{node.id}] expanded to symbolic_repr node[{newNode.id}|{symbolic_repr}]")
        return newNode

    @staticmethod
    def _parse_schema_linking(sql: str) -> List[str]:
        """解析SQL中的表和字段关系
        
        Args:
            sql: 要解析的SQL语句
            
        Returns:
            包含所有schema links的列表，格式为 ["table.field", ...]
            
        Raises:
            ValueError: 当没有找到任何schema link时
        """
        # 常量定义
        FROM_JOIN_KEYWORDS = ('FROM', 'JOIN')
        QUOTE_CHARS = ('`', '"', "'")
        
        # 解析SQL
        parsed = sqlparse.parse(sql)[0]
        table_mapping: Dict[str, str] = {}
        schema_links: List[str] = []

        def strip_quotes(value: str) -> Tuple[str, bool]:
            """去除字符串两端的引号，并返回是否包含反引号
            
            Args:
                value: 要处理的字符串
                
            Returns:
                (处理后的字符串, 是否包含反引号)
            """
            stripped = value.strip()
            has_backtick = stripped.startswith('`') and stripped.endswith('`')
            for quote in QUOTE_CHARS:
                stripped = stripped.strip(quote)
            return stripped, has_backtick

        def process_table_name(table_name: str, has_backtick: bool) -> str:
            """处理表名，根据需要添加反引号
            
            Args:
                table_name: 原始表名
                has_backtick: 是否需要添加反引号
                
            Returns:
                处理后的表名
            """
            return f"`{table_name}`" if has_backtick else table_name

        def extract_table_mapping(token: Token) -> None:
            """从 FROM/JOIN 子句中提取表名和别名的映射关系"""
            if isinstance(token, Identifier):
                parts = []
                has_backtick = False
                
                # 收集非空白token
                for t in token.tokens:
                    if t.ttype in (Name, None) and t.value.strip():
                        val, is_backtick = strip_quotes(t.value)
                        has_backtick = has_backtick or is_backtick
                        parts.append(val)
                
                if not parts:
                    return
                
                if len(parts) >= 2:  # table alias 或 table AS alias
                    real_table = process_table_name(parts[0], has_backtick)
                    alias = parts[-1]
                    table_mapping[alias] = real_table
                    # 同时添加原始表名的映射
                    if has_backtick:
                        table_mapping[real_table] = real_table
                    logging.info(f"Added mapping: {alias} -> {real_table}")
                else:  # 只有表名
                    table_name = process_table_name(parts[0], has_backtick)
                    table_mapping[table_name] = table_name
                    # 同时添加不带反引号的表名映射
                    if has_backtick:
                        stripped_name = strip_quotes(table_name)[0]
                        table_mapping[stripped_name] = table_name
                    logging.info(f"Added self mapping: {table_name} -> {table_name}")
                    
            elif token.ttype is Keyword:  # 处理表名是关键字的情况
                next_token = get_next_non_whitespace_token(token)
                if next_token and next_token.ttype in (Name, None):
                    table_name = f"`{token.value}`"
                    table_mapping[next_token.value] = table_name
                    # 同时添加原始表名的映射
                    table_mapping[table_name] = table_name
                    logging.warning(f"表名 '{token.value}' 与 SQL 关键字冲突，已添加反引号")
                    logging.info(f"Added mapping from keyword: {next_token.value} -> {table_name}")

        def get_next_non_whitespace_token(token: Token) -> Optional[Token]:
            """获取下一个非空白token"""
            parent = token.parent
            if not parent:
                return None
                
            idx = parent.token_index(token)
            for t in parent.tokens[idx + 1:]:
                if not t.is_whitespace:
                    return t
            return None

        def extract_schema_link(token: Token) -> None:
            """从token中提取schema link"""
            def add_schema_link(alias: str, field: str) -> None:
                """添加schema link到结果列表"""
                real_table = table_mapping.get(alias)
                if real_table:
                    schema_link = f"{real_table}.{field}"
                    schema_links.append(schema_link)
                    logging.info(f"Added schema link: {schema_link} (from alias: {alias})")

            if isinstance(token, Identifier):
                parts = []
                for t in token.tokens:
                    if t.ttype in (Name, None) and t.value.strip():
                        parts.append(strip_quotes(t.value)[0])
                    elif t.value.strip() == '.':
                        parts.append('.')
                
                # 检查是否形成了 table.field 模式
                if len(parts) >= 3:
                    for i in range(len(parts)-2):
                        if parts[i+1] == '.':
                            # 尝试直接使用表名
                            table_name = parts[i]
                            field = parts[i+2]
                            
                            # 如果表名在映射中存在，使用映射的表名
                            if table_name in table_mapping:
                                add_schema_link(table_name, field)
                            # 否则尝试作为别名查找
                            else:
                                add_schema_link(table_name, field)
                            break
                            
            elif token.ttype == Name:
                parent = token.parent
                if not parent:
                    return
                    
                idx = parent.token_index(token)
                if idx + 2 >= len(parent.tokens):
                    return
                    
                next_token = parent.tokens[idx + 1]
                next_next_token = parent.tokens[idx + 2]
                
                if (next_token.ttype == Punctuation and 
                    next_token.value == '.' and 
                    next_next_token.ttype == Name):
                    table_name = strip_quotes(token.value)[0]
                    field = strip_quotes(next_next_token.value)[0]
                    
                    # 尝试直接使用表名或作为别名
                    if table_name in table_mapping:
                        add_schema_link(table_name, field)
                    else:
                        add_schema_link(table_name, field)

        def get_next_table_token(token_list: TokenList, current_idx: int) -> Optional[Token]:
            """获取下一个表名 token
            
            Args:
                token_list: token列表
                current_idx: 当前token的索引
                
            Returns:
                下一个表名token，如果没有找到则返回None
            """
            for token in token_list[current_idx + 1:]:
                if isinstance(token, Identifier) or (token.ttype is Keyword and not token.is_keyword):
                    return token
                # 跳过空白字符和注释
                if token.is_whitespace or token.ttype in (sqlparse.tokens.Comment, ):
                    continue
                break
            return None

        # 第一遍：提取所有表名映射
        for token in parsed.tokens:
            if token.ttype is Keyword and token.value.upper() in FROM_JOIN_KEYWORDS:
                next_token = get_next_table_token(parsed.tokens, parsed.token_index(token))
                if next_token:
                    extract_table_mapping(next_token)
        
        logging.info(f"Final table mapping: {table_mapping}")
        
        # 第二遍：提取所有 schema links
        for token in parsed.flatten():
            logging.debug(f"Processing token: type={type(token)}, ttype={token.ttype}, value={token.value}")
            extract_schema_link(token)
        
        if not schema_links:
            raise ValueError("No schema linking found in SQL")
        
        return schema_links

    def _generate_dummy_sql_prompt(self, query: str) -> str:
        """生成dummy_sql提示"""
        return escape(f"""
        [{self.config.model.sql_model}] 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决SQL
        schema及问题 ```{query}```

        注意：
        （1）请不要使用不存在的表和字段
        （2）表名请加反引号，防止表名和字段名与SQL关键字冲突
        
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

        注意：
        （1）不是用 pandas 代码调用 sql，而是直接用 pandas 表达业务逻辑

        结果 pandas 代码如下：
        """)

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    def check_schema_links(actual_links: list[str], expected_links: list[str]):
        """检查 schema links 是否符合预期"""
        # 对结果排序以确保比较的一致性
        actual_sorted = sorted(actual_links)
        expected_sorted = sorted(expected_links)
        
        assert actual_sorted == expected_sorted, \
            f"Schema links 不匹配！\n预期: {expected_sorted}\n实际: {actual_sorted}"
        print("✓ 测试通过")
    
    # 测试用例1：简单的表别名
    sql1 = """
    SELECT d.A11 AS average_income, a.type AS account_type
    FROM district d
    JOIN `account` a ON d.district_id = a.district_id
    WHERE d.A11 > 8000
    """
    print("\n测试用例1 - 简单的表别名:")
    print("SQL:", sql1)
    result1 = TaskAlignmentSearcher._parse_schema_linking(sql1)
    print("解析结果:", result1)
    expected1 = [
        'district.A11',
        '`account`.type',
        'district.district_id',
        '`account`.district_id',
        'district.A11'
    ]
    check_schema_links(result1, expected1)
    
    # 测试用例2：使用 AS 关键字的表别名
    sql2 = """
    SELECT c.client_id, t.trans_id
    FROM client AS c
    JOIN `transaction` AS t ON c.client_id = t.client_id
    WHERE t.amount > 1000
    """
    print("\n测试用例2 - 使用 AS 关键字:")
    print("SQL:", sql2)
    result2 = TaskAlignmentSearcher._parse_schema_linking(sql2)
    print("解析结果:", result2)
    expected2 = [
        'client.client_id',
        '`transaction`.trans_id',
        'client.client_id',
        '`transaction`.client_id',
        '`transaction`.amount'
    ]
    check_schema_links(result2, expected2)
    
    # 测试用例3：复杂查询
    sql3 = """
    SELECT DISTINCT d.A11 AS average_income, 
           a.type AS account_type,
           l.amount AS loan_amount
    FROM district d
    JOIN `account` a ON d.district_id = a.district_id
    JOIN disp dp ON a.account_id = dp.account_id
    JOIN client c ON dp.client_id = c.client_id
    JOIN loan l ON a.account_id = l.account_id
    WHERE d.A11 > 8000 AND d.A11 <= 9000
    AND a.type != 'OWNER'
    """
    print("\n测试用例3 - 复杂查询:")
    print("SQL:", sql3)
    result3 = TaskAlignmentSearcher._parse_schema_linking(sql3)
    print("解析结果:", result3)
    expected3 = [
        'district.A11',
        '`account`.type',
        'loan.amount',
        'district.district_id',
        '`account`.district_id',
        '`account`.account_id',
        'disp.account_id',
        'disp.client_id',
        'client.client_id',
        '`account`.account_id',
        'loan.account_id',
        'district.A11',
        'district.A11',
        '`account`.type'
    ]
    check_schema_links(result3, expected3)

    sql4 = """
    SELECT DISTINCT `account`.`frequency` AS `account_type`
    FROM `account`
    JOIN `district` ON `account`.`district_id` = `district`.`district_id`
    WHERE `account`.`frequency` != 'OWNER'
    AND `district`.`A11` > 8000
    AND `district`.`A11` <= 9000;
    """
    print("\n测试用例4 - 复杂查询:")
    print("SQL:", sql4)
    result4 = TaskAlignmentSearcher._parse_schema_linking(sql4)
    print("解析结果:", result4)
    expected4 = [
        '`account`.frequency',
        '`account`.district_id',
        '`district`.district_id',
        '`account`.frequency',
        '`district`.A11',
        '`district`.A11'
    ]
    check_schema_links(result4, expected4)
    
    print("\n所有测试用例均通过！")