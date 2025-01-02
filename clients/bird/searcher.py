from typing import Any, Optional
from milkie.sdk.agent_client import AgentClient
from milkie.sdk.config_server import ConfigServer
from clients.bird.config import Config
from clients.bird.database import Database
from clients.bird.logger import INFO, ERROR
from clients.bird.tree import SearchTree, Node, NodeType
from milkie.utils.data_utils import escape

class Searcher:
    def __init__(self, 
            config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config
        self._client = AgentClient(ConfigServer(config.agent.addr))
        self._db = Database(
            config.database,
            self._client
        )
        self.agent_name = config.agent.name

    def thought(self, query: str, error_patterns: set, trial: int) -> str:
        error_hints = ""
        if error_patterns:
            error_hints = "\n已有的错误模式：\n" + "\n".join(f"- {e}" for e in error_patterns)
            
        return escape(f"""
    [{self._get_thought_model()}] (trial: {trial}) 请根据请求中包括的 schema、问题做分析，一步一步思考，给出问题的解决思路
    schema及问题 ```{query}```
    可用schema解释 ```{self._db.descTablesFromQuery(query)}```
    {error_hints}

    请注意：
    1. 首先明确 query 中的问题问的 metric，不需要回答多余的信息，例如问"人口最多的城市是哪个"，只需要回答满足要求的城市，而不需要返回人口数量
    2. 分析需要用到的表和字段，仅使用必要的表和字段
    3. 考虑表之间的关联关系
    4. 如果有错误模式，思考如何避免这些错误
    
    现在请输出你的分析和思考，请不要直接输出 sql：
""")

    def sql(self, query: str, thought: str, error_patterns: set, trial: int) -> str:
        error_hints = ""
        if error_patterns:
            error_hints = "\n已有的错误模式：\n" + "\n".join(f"- {e}" for e in error_patterns)
            
        return escape(f"""
    [{self.config.model.sql_model}] (trial: {trial}) 请结合原始问题和分析思考结果，给出最终的 sql 
    请注意以下规则：
    1. 仔细检查 tables 的 schema，不要使用不存在的 column
    2. SQL 中不允许直接在 MAX/MIN 中嵌套 SUM 函数
    3. 如果有错误模式，必须避免这些错误，确保表的连接条件正确
    4. 输出的 SQL 必须是完整的、可执行的
    5. 请看清楚问题所需要问的信息，回答且仅回答必要的信息

    schema及问题 ```{query}```
    分析思考结果 ```{thought}```
    {error_hints}
    
    现在请输出最终 sql：
""")

    def forward_step(self, tree: SearchTree):
        """执行一次前向扩展，采用广度优先策略"""
        # 当前层级的所有节点
        current_level = []
        next_level = []
        
        # 如果栈为空，说明是初始状态，从根节点开始
        if not tree.stack:
            if not tree.root.is_completed():
                current_level.append(tree.root)
        else:
            current_level = tree.stack[:]
            
        tree.stack = []  # 清空栈，准备收集下一层级的节点
        
        # 处理当前层级的所有节点
        for node in current_level:
            if node.is_completed():
                continue
                
            if node.type == NodeType.ROOT:
                # ROOT节点只产生THOUGHT节点
                if node.should_continue_thought(self._get_max_thoughts()):
                    try:
                        trial = node.increment_thought_count()
                        code = self.thought(
                            node.expa.query,
                            node.expa.error_patterns,
                            trial - 1
                        )
                        thought = self._client.execute(code, self.agent_name)
                                                
                        thought_node = Node(
                            type=NodeType.THOUGHT,
                            parent=node,
                            children=[],
                            depth=node.depth + 1
                        )
                        thought_node.expa.query = node.expa.query
                        thought_node.expa.thought = thought
                        thought_node.expa.error_patterns = set()
                        
                        node.children.append(thought_node)
                        next_level.append(thought_node)
                        INFO(f"Node[{node.id}] expanded to THOUGHT node[{thought_node.id}|{self._unnewline(thought)}]")
                    except Exception as e:
                        ERROR(f"Node[{node.id}] failed to expand to THOUGHT: {str(e)}")
                    
            elif node.type == NodeType.THOUGHT:
                # THOUGHT节点产生SQL节点
                if node.should_continue_sql(self.config.search.min_sqls, self.config.search.max_sqls):
                    try:
                        trial = node.increment_sql_count()
                        code = self.sql(
                            node.expa.query,
                            node.expa.thought,
                            node.expa.error_patterns,
                            trial - 1
                        )
                        sql = self._client.execute(code, self.agent_name)
                        sql = self._preprocess_sql(sql)
                        result, error = self._db.execsql(sql)
                        success = result is not None
                        
                        sql_node = Node(
                            type=NodeType.SQL,
                            parent=node,
                            children=[],
                            depth=node.depth + 1
                        )
                        sql_node.other.sql = sql
                        sql_node.other.result = result
                        sql_node.other.success = success
                        
                        if not success:
                            sql_node.other.error = error
                            node.add_error_pattern(error)
                        else:
                            node.increment_success_count()
                        
                        node.children.append(sql_node)
                        tree.leaf_nodes.append(sql_node)
                        
                        if success:
                            INFO(f"Node[{node.id}] expanded to successful SQL node[{sql_node.id}|{self._unnewline(sql)}] with result: {result}")
                        else:
                            INFO(f"Node[{node.id}] expanded to failed SQL node[{sql_node.id}|{self._unnewline(sql)}]")
                            
                    except Exception as e:
                        error_msg = str(e)
                        ERROR(f"Node[{node.id}] failed to expand to SQL: {error_msg}")
                        node.add_error_pattern(error_msg)
                        
            elif node.type == NodeType.SQL:
                # SQL 节点是叶子节点，不需要扩展
                if node not in tree.leaf_nodes:
                    tree.leaf_nodes.append(node)
                    
        # 检查当前层级节点是否需要继续探索
        for node in current_level:
            if node.should_mark_completed(
                self.config.search.min_sqls,
                self.config.search.max_sqls,
                self._get_max_thoughts()
            ):
                node.mark_completed()
            elif not node.is_completed():
                next_level.append(node)
                    
        # 更新下一轮要处理的节点
        tree.stack = next_level

    def inference(self, query: str) -> str:
        try:
            # 创建搜索树
            tree = SearchTree(query, self.config.search.max_iters)
            
            # 执行前向扩展直到完成
            while not tree.is_completed():
                self.forward_step(tree)
                tree.iteration += 1
            
            # 选择最佳 SQL
            return tree.get_best_sql()
            
        except Exception as e:
            ERROR(f"Error in inference: {str(e)}")
            raise

    def _get_thought_model(self) -> str:
        return self.config.model.thought_model

    def _get_max_thoughts(self) -> int:
        return self.config.search.max_thoughts

    def _preprocess_sql(self, sql: str) -> str:
        if sql.startswith("```sql") and sql.endswith("```"):
            return sql[6:-3]
        elif sql.startswith("```mysql") and sql.endswith("```"):
            return sql[8:-3]
        return sql

    def _unnewline(self, sql: str) -> str:
        return sql.replace(chr(10), "|")