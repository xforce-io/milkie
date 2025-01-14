from typing import Optional, Set
from clients.bird.base_searcher import BaseSearchTree, Node, NodeType, NodeExpansionRule
from clients.bird.config import Config
from clients.bird.database import Database
from clients.bird.logger import INFO, ERROR
from milkie.sdk.agent_client import AgentClient
from milkie.sdk.config_server import ConfigServer
from milkie.utils.data_utils import escape

class BaseSqlNodeType(NodeType):
    """BaseSQL特定的节点类型"""
    SQL = "sql"

class BaseSqlSearcher(BaseSearchTree):
    def __init__(self, 
            query: str,
            max_iters: int,
            config: Optional[Config] = None):
        if config is None:
            config = Config.load()
            
        self.config = config
        self.query = query
        self._client = AgentClient(ConfigServer(config.agent.addr))
        self._db = Database(
            config.database,
            self._client
        )
        
        super().__init__(max_iters)
        
    def _expand_sql(self, node: Node, cot :str) -> Node:
        """扩展SQL节点"""
        code = self._generate_sql_prompt(
            self.query,
            cot,
            node.get_error_patterns(),
            node.get_num_children()
        )
        sql = self._client.execute(code, self.config.agent.name)
        sql = self._preprocess_sql(sql)
        result, error = self._db.execsql(sql)
        
        sql_node = Node(
            type=BaseSqlNodeType.SQL,
            parent=node,
            depth=node.depth + 1
        )
        sql_node.data["sql"] = sql
        sql_node.data["result"] = result
        sql_node.data["success"] = result is not None
        
        if not sql_node.data["success"]:
            sql_node.data["error"] = error
            sql_node.add_error_pattern(error)
        else:
            node.add_successful_child()
            # 如果是第一次就成功，标记为高置信度
            if node.get_num_children() == 1:
                sql_node.high_confidence = True
            
        if sql_node.data["success"]:
            INFO(f"Node[{node.id}] expanded to successful SQL node[{sql_node.id}|{self._unnewline(sql)}] with result: {result}")
        else:
            INFO(f"Node[{node.id}] expanded to failed SQL node[{sql_node.id}|{self._unnewline(sql)}]")
            
        return sql_node
        
    def _process_expansion_result(self, source_node: Node, new_node: Node):
        """处理扩展结果"""
        source_node.add_child(new_node)
        if new_node.type == BaseSqlNodeType.SQL:
            self.leaf_nodes.append(new_node)
           
    def get_best_sql(self) -> str:
        """获取最佳SQL结果"""
        successful_nodes = [
            node for node in self.leaf_nodes 
            if node.type == BaseSqlNodeType.SQL and 
            node.data["success"]
        ]
        
        if not successful_nodes:
            raise RuntimeError("No successful SQL found")
            
        # 这里可以添加更复杂的选择逻辑
        return successful_nodes[-1].data["sql"]
        
    def _generate_sql_prompt(self, query: str, thought: str, error_patterns: Set[str], trial: int) -> str:
        """生成SQL提示"""
        error_hints = ""
        if error_patterns:
            error_hints = "\n已有的错误模式：\n" + "\n".join(f"- {e}" for e in error_patterns)
            
        if self.config.search.table_fields_record_samples > 0:
            schema_desc_prompt = f'''可用schema解释 ```{self._db.descTableFieldsFromQuery(query, self.config.search.table_fields_record_samples)}```'''
        else:
            schema_desc_prompt = ""
            
        return escape(f"""
    [{self.config.model.sql_model}] (trial: {trial}) 请结合原始问题和分析思考结果，给出最终的 sql 
    schema及问题 ```{query}```
    {schema_desc_prompt}
    分析思考结果 ```{thought}```
    已有错误模式 ```{error_hints}```
    
    请注意以下规则：
    1. 仅输出单条 SQL，请保证输出的 SQL 必须是完整的、可执行的
    2. 仔细检查 tables 的 schema，不要使用不存在的 column，和 mysql 关键词冲突的 column 请进行转义,比如 Virtual 转成 `Virtual`
    3. SQL 中不允许直接在 MAX/MIN 中嵌套 SUM 函数

    现在请输出最终这条 sql：
""")
        
    def _preprocess_sql(self, sql: str) -> str:
        """预处理SQL语句"""
        if sql.startswith("```sql") and sql.endswith("```"):
            return sql[6:-3]
        elif sql.startswith("```mysql") and sql.endswith("```"):
            return sql[8:-3]
        return sql
        
    def _unnewline(self, text: str) -> str:
        """移除换行符"""
        return text.replace(chr(10), "|") 
        
    def inference(self) -> str:
        """执行搜索并返回最佳SQL"""
        try:
            # 执行前向扩展直到完成
            while not self.is_completed():
                self.forward_step()
                self.iteration += 1
            
            # 选择最佳SQL
            return self.get_best_sql()
            
        except Exception as e:
            ERROR(f"Error in inference: {str(e)}")
            raise 