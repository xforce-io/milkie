import pymysql
import re
from typing import Optional, Union, Dict, List, Set, Tuple, Any
from clients.bird.config import Config, DatabaseConfig
from clients.bird.logger import logger, INFO, ERROR
from milkie.sdk.agent_client import AgentClient
from milkie.cache.cache_kv import CacheKVMgr
from milkie.utils.data_utils import escape
import os

class Database:
    # 类级别的缓存管理器
    _cache_mgr = None
    
    # 配置常量
    SAMPLE_SIZE = 30  # 每个字段采样数量
    
    def __init__(
            self, 
            config: Optional[DatabaseConfig] = None,
            client: Optional[AgentClient] = None):
        if config is None:
            config = Config.load().database
        self._config = config  # 保存配置以便重连
        self._client = client
        self._connect()
        
        # 确保缓存管理器已初始化
        self._init_cache()
        
    def execsql(self, sql: str) -> tuple[Optional[str], Optional[str]]:
        """
        执行SQL查询
        
        Returns:
            tuple[Optional[str], Optional[str]]: 
                - 非空结果字符串
                - 空结果字符串
                - 执行出错
        """
        cursor = None
        try:
            # 检查连接是否断开，如果断开则重连
            self._db.ping(reconnect=True)
            
            cursor = self._db.cursor()
            cursor.execute(sql)
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # 获取结果
            results = cursor.fetchall()
            
            # 格式化结果
            formatted_results = []
            for row in results:
                formatted_row = dict(zip(columns, row))
                formatted_results.append(formatted_row)
            
            return str(formatted_results), None
            
        except Exception as e:
            logger.error(f"Error executing SQL: {sql}, error: {str(e)}")
            return None, str(e)
        finally:
            if cursor:
                cursor.close()
                
    def descTablesFromQuery(self, query: str) -> List[str]:
        """
        从查询中提取表名并获取每个表的描述
        
        Args:
            query: 包含建表语句的查询字符串
            
        Returns:
            List[str]: 每个表的描述列表
        """
        try:
            # 1. 提取表名
            table_names = self._extract_table_names(query)
            if not table_names:
                ERROR("No tables found in query")
                return []
                
            # 2. 获取每个表的描述
            table_briefs = []
            for table_name in table_names:
                brief = self._descTable(table_name)
                if brief:
                    table_briefs.append(brief)
                    
            return table_briefs
            
        except Exception as e:
            ERROR(f"Error in descTablesFromQuery: {str(e)}")
            return []
            
    def descTableFieldsFromQuery(self, query: str) -> Dict[str, str]:
        """
        生成表和字段的详细描述
        
        Args:
            query: SQL查询字符串
            
        Returns:
            Dict[str, str]: 表名到描述的映射
        """
        try:
            # 1. 从缓存中检查
            cache_key = [{"query": query}]
            cached_desc = self._cache_mgr.getValue("tablefields", cache_key)
            if cached_desc:
                INFO(f"Cache hit for query")
                return cached_desc
            
            # 2. 提取表名和字段
            table_fields = self._extract_tables_and_fields(query)
            if not table_fields:
                ERROR("No tables or fields found in query")
                return {}
                
            # 3. 验证表和字段的存在性，获取有效的表字段映射
            valid_table_fields = self._validate_tables_and_fields(table_fields)
            if not valid_table_fields:
                ERROR("No valid tables or fields found")
                return {}
                
            # 4. 获取字段样本数据并生成描述
            descriptions = {}
            for table_name, fields in valid_table_fields.items():
                desc = self._generate_field_description(table_name, fields)
                if desc:
                    descriptions[table_name] = desc
                    
            # 5. 存入缓存
            if descriptions:
                self._cache_mgr.setValue("tablefields", cache_key, descriptions)
                
            return descriptions
            
        except Exception as e:
            ERROR(f"Error in descTableFields: {str(e)}")
            return {}
            
    @classmethod
    def _init_cache(cls):
        """初始化缓存管理器（如果还未初始化）"""
        if cls._cache_mgr is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '../../data/cache')
            cls._cache_mgr = CacheKVMgr(cache_dir, dumpInterval=5, expireTimeByDay=30)
    
    def _connect(self):
        """重新建立数据库连接"""
        self._db = pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            read_timeout=15,
            write_timeout=15
        )
        
    def _extract_tables_and_fields(self, query: str) -> Dict[str, Set[str]]:
        """
        从查询中提取表名和字段名
        
        Args:
            query: SQL查询字符串
            
        Returns:
            Dict[str, Set[str]]: 表名到字段集合的映射
        """
        # 提取所有表名
        tables = set(self._extract_table_names(query))
        if not tables:
            return {}
            
        # 提取字段名（包括带表名前缀和不带前缀的）
        field_pattern = r'(?:(`?\w+`?)\.)?(`?\w+`?)'
        matches = re.finditer(field_pattern, query)
        
        # 初始化结果字典
        result = {table: set() for table in tables}
        
        # 处理每个匹配
        for match in matches:
            table_name, field_name = match.groups()
            if table_name:  # 带表名前缀的字段
                table_name = table_name.strip('`')
                if table_name in tables:
                    result[table_name].add(field_name.strip('`'))
            else:  # 不带表名前缀的字段
                field_name = field_name.strip('`')
                # 将字段添加到所有表中，后续验证时会过滤掉无效的
                for table in tables:
                    result[table].add(field_name)
                    
        return result
        
    def _validate_tables_and_fields(self, table_fields: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        验证表和字段是否存在
        
        Args:
            table_fields: 表名到字段集合的映射
            
        Returns:
            Dict[str, Set[str]]: 验证后的表名到字段集合的映射
        """
        valid_mappings = {}
        cursor = None
        try:
            cursor = self._db.cursor()
            
            for table_name, fields in table_fields.items():
                try:
                    # 获取表的所有字段
                    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                    valid_fields = {row[0] for row in cursor.fetchall()}
                    
                    # 过滤出有效字段
                    valid_table_fields = fields & valid_fields
                    if valid_table_fields:
                        valid_mappings[table_name] = valid_table_fields
                        
                except Exception as e:
                    ERROR(f"Error validating table {table_name}: {str(e)}")
                    continue
                    
            return valid_mappings
            
        except Exception as e:
            ERROR(f"Error in validate_tables_and_fields: {str(e)}")
            return {}
        finally:
            if cursor:
                cursor.close()
                
    def _get_field_samples(self, table_name: str, fields: Set[str]) -> Dict[str, List[Any]]:
        """
        获取字段的样本数据
        
        Args:
            table_name: 表名
            fields: 字段集合
            
        Returns:
            Dict[str, List[Any]]: 字段到样本值列表的映射
        """
        cursor = None
        try:
            cursor = self._db.cursor()
            
            # 构建查询语句
            fields_str = ", ".join(f"DISTINCT {field}" for field in fields)
            query = f"SELECT {fields_str} FROM {table_name} LIMIT {self.SAMPLE_SIZE}"
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # 整理结果
            samples = {field: [] for field in fields}
            for row in results:
                for i, field in enumerate(fields):
                    if row[i] is not None:  # 排除空值
                        samples[field].append(row[i])
                        
            return samples
            
        except Exception as e:
            ERROR(f"Error getting field samples for table {table_name}: {str(e)}")
            return {}
        finally:
            if cursor:
                cursor.close()
                
    def _get_field_desc_prompt(self, table_name: str, fields: Set[str], schema_info: List[str], samples: Dict[str, List[Any]]) -> str:
        """生成字段描述的prompt"""
        return escape(f"""
请分析下面这张数据库表的字段信息和样本数据，给出详细的字段描述。

表名：{table_name}

字段结构：
{chr(10).join(schema_info)}

字段样本值：
{str(samples)}

请按照以下格式输出分析结果：
1. 表的简要说明：
<简要描述表的主要用途>

2. 字段详细说明：
<针对每个字段，详细说明：
- 字段的具体用途
- 数据类型及其限制
- 可能的取值范围或枚举值
- 基于样本数据的特征分析>
""")
        
    def _generate_field_description(self, table_name: str, fields: Set[str]) -> Optional[str]:
        """
        生成字段的详细描述
        
        Args:
            table_name: 表名
            fields: 字段集合
            
        Returns:
            Optional[str]: 生成的描述
        """
        cursor = None
        try:
            cursor = self._db.cursor()
            
            # 1. 获取字段的schema信息
            cursor.execute(f"SHOW FULL COLUMNS FROM {table_name}")
            columns = cursor.fetchall()
            
            # 过滤出目标字段的schema信息
            schema_info = []
            for col in columns:
                if col[0] in fields:
                    field_name = col[0]
                    field_type = col[1]
                    field_comment = col[8] if len(col) > 8 else ''
                    schema_info.append(f"{field_name} {field_type} - {field_comment}")
                    
            # 2. 获取样本数据
            samples = self._get_field_samples(table_name, fields)
            if not samples:
                return None
                
            # 3. 生成描述
            if self._client:
                prompt = self._get_field_desc_prompt(table_name, fields, schema_info, samples)
                description = self._client.execute(prompt, "cot_expert")
                return description
                
            return None
            
        except Exception as e:
            ERROR(f"Error generating field description for table {table_name}: {str(e)}")
            return None
        finally:
            if cursor:
                cursor.close()
                
    def _extract_table_names(self, query: str) -> List[str]:
        """
        从查询中提取表名
        
        Args:
            query: 包含建表语句的查询字符串
            
        Returns:
            List[str]: 表名列表
        """
        # 使用正则表达式匹配 CREATE TABLE 语句中的表名
        pattern = r"CREATE\s+TABLE\s+(?:`?(\w+)`?)"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        return [match.group(1) for match in matches]
    
    def __del__(self):
        if hasattr(self, '_db'):
            self._db.close()
