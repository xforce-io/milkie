from abc import abstractmethod
from typing import Any, Dict, List, Optional
import logging
import re
import datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# 添加 NullPool 导入
from sqlalchemy.pool import NullPool

from .datasource import DataSource
from milkie.config.config import DataSourceType
from milkie.ontology.basic.concept import Concept, ConceptMemberType
from milkie.ontology.mapping import Mapping

logger = logging.getLogger(__name__)


def _camelCase(s: str) -> str:
    """将下划线或空格分隔的字符串转换为驼峰式"""
    s = re.sub(r"[_\-]+", " ", s).title().replace(" ", "")
    return s[0].lower() + s[1:] if s else ""


class DataSourceSql(DataSource):
    """SQL 类型数据源的基类，使用 SQLAlchemy"""

    def __init__(self, name: str, type: DataSourceType, config: Dict[str, Any]):
        super().__init__(name, type, config)
        self._engine: Engine = None
        self._inspector = None
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 3306)  # 默认 MySQL 端口
        self.username = config.get("username")
        self.password = config.get("password")
        self.database = config.get("database")

    @property
    def type(self) -> DataSourceType:
        return self._type # 返回存储的类型

    @abstractmethod
    def connect(self) -> Engine:
        """建立数据库连接"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """获取数据库模式 (表名 -> 列信息列表, 每个列信息为 {'name':列名, 'type':类型字符串})"""
        pass

    def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            try:
                self._engine.dispose()
                logger.info(f"数据库连接已关闭: {self.name}")
                self._engine = None
            except Exception as e:
                logger.error(f"关闭数据库连接时出错 {self.name}: {e}")
        else:
            logger.warning(f"尝试关闭一个未建立或已关闭的连接: {self.name}")

    def test_connection(self) -> bool:
        """测试数据库连接是否成功"""
        originalConnectionState = self._engine
        connToClose = None
        try:
            if not self._engine:
                connToClose = self.connect()
            if not self._engine:
                logger.warning(f"测试连接 {self.name} 失败：无法建立连接")
                return False
            # 简单的测试查询
            self.executeQuery("SELECT 1", fetchColumns=False) # test_connection 不关心结果
            logger.info(f"测试连接 {self.name} 成功")
            return True
        except Exception as e:
            logger.error(f"测试连接失败 {self.name}: {e}")
            return False
        finally:
            # 如果是为了测试而临时建立的连接，则关闭它
            if connToClose and connToClose == self._engine:
                self.close()
            # 恢复原始连接状态（如果测试前就有连接）
            elif originalConnectionState and not self._engine:
                self._engine = originalConnectionState # 避免影响后续操作

    # 可以添加 SQL 特有的方法，例如执行 SQL 语句
    def executeQuery(self, query: str, fetchColumns: bool = True) -> Dict[str, Any]:
        """执行一个 SQL 查询并返回结果

        Args:
            query (str): 要执行的 SQL 查询语句
            fetchColumns (bool): 是否获取列名信息，默认为 True

        Returns:
            Dict[str, Any]: 包含查询结果和列名（如果 fetchColumns 为 True）
        """
        conn = self._engine
        shouldCloseConn = False
        if not conn:
            conn = self.connect()
            if not conn:
                raise ConnectionError(f"无法连接到数据库: {self.name}")
            shouldCloseConn = True # 如果是临时连接，用完要关闭

        cursor = None
        try:
            cursor = conn.connect().execute(text(query))
            results = cursor.fetchall()
            if fetchColumns:
                # Use cursor.keys() to get column names from SQLAlchemy's Result object
                # The Result object (cursor) itself doesn't have a 'description' attribute directly
                columns = list(cursor.keys())
                return {"columns": columns, "data": results}
            return {"columns": [], "data": results} # Return empty columns if not fetching
        except Exception as e:
            logger.error(f"执行查询时出错 on {self.name}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if shouldCloseConn and conn:
                conn.dispose()
                if conn == self._engine:
                    self._engine = None # 确保内部状态一致

    def _map_db_type_to_concept_type(self, db_type_full: str) -> ConceptMemberType:
        """将数据库特定的列类型字符串映射到 ConceptMemberType。"""
        logger.debug(f"Mapping DB type: Original='{db_type_full}', Type='{type(db_type_full)}'")
        try:
            if not db_type_full or not isinstance(db_type_full, str): # Ensure it's a non-empty string
                logger.warning(f"Invalid db_type_full: '{db_type_full}'. Defaulting to ANY.")
                return ConceptMemberType.ANY

            # 提取括号前的主要类型部分，并转为小写
            db_type_main = db_type_full.split('(')[0].strip().lower()

            result_type = ConceptMemberType.ANY # Default

            if "char" in db_type_main or \
               "varchar" in db_type_main or \
               "text" in db_type_main or \
               "string" in db_type_main or \
               "enum" in db_type_main or \
               "set" in db_type_main:
                result_type = ConceptMemberType.STRING
            elif "int" in db_type_main or \
                 "integer" in db_type_main or \
                 "tinyint" in db_type_main or \
                 "smallint" in db_type_main or \
                 "mediumint" in db_type_main or \
                 "bigint" in db_type_main:
                result_type = ConceptMemberType.NUMBER
            elif "float" in db_type_main or \
                 "double" in db_type_main or \
                 "decimal" in db_type_main or \
                 "numeric" in db_type_main or \
                 "real" in db_type_main:
                result_type = ConceptMemberType.NUMBER
            elif "bool" in db_type_main or "boolean" in db_type_main:
                result_type = ConceptMemberType.BOOLEAN
            elif "date" in db_type_main or \
                 "datetime" in db_type_main or \
                 "timestamp" in db_type_main or \
                 "time" in db_type_main or \
                 "year" in db_type_main:
                result_type = ConceptMemberType.STRING # Or a more specific date/time type if available
            elif "json" in db_type_main:
                result_type = ConceptMemberType.STRING # Or OBJECT if handling structured JSON

            if result_type == ConceptMemberType.ANY and db_type_main not in ['unknown', '']: # Log if no specific mapping found, unless it was already 'unknown' or empty
                logger.warning(f"Unknown DB type: '{db_type_full}' (main: '{db_type_main}'), defaulted to ANY.")

            logger.debug(f"Mapped DB type '{db_type_full}' to ConceptMemberType '{result_type.name if result_type else 'None'}' (Python type: {type(result_type)})")
            return result_type
        except Exception as e:
            logger.error(f"Error mapping DB type '{db_type_full}': {e}. Defaulting to ANY.", exc_info=True)
            return ConceptMemberType.ANY

    def scan(self) -> List[Mapping]:
        """扫描 SQL 数据库 schema，为每个表生成 Concept 和 Mapping"""
        logger.info(f"开始扫描数据源: {self.name}")
        mappings = []
        try:
            schema = self.get_schema()
            if not schema:
                logger.warning(f"无法获取数据源 {self.name} 的 schema 信息")
                return []

            for table_name, columns_details in schema.items():
                # 1. 创建 Concept
                concept_name = _camelCase(table_name).capitalize()
                
                members = {}
                valid_columns_for_mapping = []
                if not columns_details:
                    logger.warning(f"表 '{table_name}' 在 {self.name} 中没有列信息，跳过")
                    continue

                for col_detail in columns_details:
                    col_name = col_detail.get('name')
                    col_type_str = col_detail.get('type')

                    if not col_name: # 跳过没有列名的条目
                        logger.debug(f"Skipping column with no name in table '{table_name}'. Detail: {col_detail}")
                        continue
                    
                    member_name = _camelCase(col_name)
                    member_type = self._map_db_type_to_concept_type(col_type_str)
                    members[member_name] = member_type
                    valid_columns_for_mapping.append(col_name)

                if not members:
                    logger.warning(f"表 '{table_name}' 在 {self.name} 中没有可转换为成员的有效列，跳过创建 Concept")
                    continue
                
                try:
                    concept = Concept(name=concept_name, members=members)
                    logger.debug(f"为表 '{table_name}' 创建了 Concept: {concept_name} with members: {members}")
                except ValueError as e:
                    logger.error(f"为表 '{table_name}' 创建 Concept '{concept_name}' 失败: {e}")
                    continue

                # 2. 创建 Mapping
                # 字段到成员的映射：列名 -> 驼峰式成员名
                # 使用 valid_columns_for_mapping 来确保只映射实际存在的列
                fieldToMemberMap = {col: _camelCase(col) for col in valid_columns_for_mapping}
                if not fieldToMemberMap: # 理论上如果 members 有内容，这里也应该有
                    logger.warning(f"表 '{table_name}' 在 {self.name} 中没有有效列名可映射，跳过创建 Mapping")
                    continue

                try:
                    mapping = Mapping(
                            dataSource=self, 
                            space=table_name,
                            concept=concept, 
                            fieldToMemberMap=fieldToMemberMap)
                    mappings.append(mapping)
                    logger.debug(f"为 Concept '{concept_name}' 创建了 Mapping")
                except (ValueError, TypeError) as e:
                    logger.error(f"为 Concept '{concept_name}' 创建 Mapping 失败: {e}")
           
            logger.info(f"数据源 {self.name} 扫描完成，生成了 {len(mappings)} 个 Mappings")
            return mappings

        except ConnectionError as e:
            logger.error(f"扫描数据源 {self.name} 失败：连接错误 {e}")
            return []
        except Exception as e:
            logger.error(f"扫描数据源 {self.name} 时发生意外错误: {e}")
            # 可以考虑抛出异常或返回空列表
            return []


class DataSourceMysql(DataSourceSql):
    """MySQL 数据源实现"""

    def __init__(self, name: str, config: Dict[str, Any]):
        # 直接传递正确的类型 DataSourceType.MYSQL
        super().__init__(name, DataSourceType.MYSQL, config)
        self._type = DataSourceType.MYSQL # 存储具体类型

    def connect(self) -> Engine:
        """连接到 MySQL 数据库"""
        if self._engine:
            logger.debug(f"已存在到 {self.name} 的连接，将重新连接")
            self.close()

        try:
            # 使用 mysql-connector-python 连接 MySQL 数据库
            # 确保已安装：pip install mysql-connector-python
            import mysql.connector # 移到这里，只有需要时才导入
            
            # 修改连接方式，为测试环境禁用连接池
            connection_url = f"mysql+mysqlconnector://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            
            # 使用 NullPool 禁用连接池，或者使用更合适的连接池配置
            # 在测试环境中，禁用连接池通常更简单可靠
            self._engine = create_engine(
                connection_url,
                poolclass=NullPool,  # 禁用连接池
            )
            
            self._inspector = inspect(self._engine)
            logger.info(f"成功连接到 MySQL 数据库: {self.name}")
            return self._engine
        except ImportError:
            logger.error(f"连接 MySQL 数据库 {self.name} 失败：缺少 'mysql-connector-python' 库。请运行 'pip install mysql-connector-python'")
            self._engine = None
            raise ConnectionError(f"缺少 MySQL 驱动: {self.name}")
        except mysql.connector.Error as err:  # 捕捉特定数据库连接错误
            logger.error(f"连接 MySQL 数据库失败 {self.name}: {err}")
            self._engine = None
            raise ConnectionError(f"无法连接到 MySQL 数据库: {self.name}, {err}")
        except Exception as e: # 其他意外错误
            logger.error(f"连接 MySQL 时发生未知错误 {self.name}: {e}")
            self._engine = None
            raise ConnectionError(f"连接 MySQL 时发生未知错误: {self.name}, {e}")

    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """获取 MySQL 数据库的模式信息 (表名 -> 列信息列表)"""
        if self.schema:
            return self.schema

        conn = self._engine
        should_close_conn = False
        if not conn:
            conn = self.connect() # 尝试连接
            if not conn:
                raise ConnectionError(f"无法获取模式，数据库未连接: {self.name}")
            should_close_conn = True

        schema: Dict[str, List[Dict[str, str]]] = {}
        cursor = None
        try:
            # 使用 self._inspector 来获取表名，更符合 SQLAlchemy 的方式
            if not self._inspector: # 确保 inspector 存在
                 if not conn: # 如果 conn 之前没有成功建立
                    conn = self.connect()
                    if not conn:
                        raise ConnectionError(f"无法获取模式，数据库未连接: {self.name}")
                 self._inspector = inspect(conn)

            tables = self._inspector.get_table_names()
            
            # 获取每个表的列名和类型
            for table_name in tables:
                # 使用 self._inspector 获取列信息
                columns_info = self._inspector.get_columns(table_name)
                
                current_table_cols = []
                for column_data in columns_info:
                    # column_data 是一个字典，包含 'name', 'type', 'nullable', 'default' 等键
                    # 'type' 通常是 SQLAlchemy 类型对象，需要转换为字符串
                    col_name = column_data.get('name')
                    col_type_obj = column_data.get('type')
                    
                    if col_name and col_type_obj is not None: # 确保列名和类型存在
                        # 将 SQLAlchemy 类型对象转换为字符串表示形式
                        # 例如：VARCHAR(length=50), INTEGER(), NUMERIC(precision=10, scale=2)
                        col_type_str = str(col_type_obj)
                        current_table_cols.append({'name': col_name, 'type': col_type_str})
                    elif col_name: # 类型未知，但列名存在
                        current_table_cols.append({'name': col_name, 'type': 'UNKNOWN'}) # 或者记录一个默认值
                        logger.warning(f"Column '{col_name}' in table '{table_name}' has an unknown type.")
                
                if current_table_cols: # 只有当表有列时才添加到 schema
                    schema[table_name] = current_table_cols
                else:
                    logger.info(f"Table '{table_name}' has no columns or columns could not be retrieved.")
           
            logger.debug(f"获取到 {self.name} 的 schema: {len(schema)} 个表")
            self.schema = schema
            return schema
        except Exception as err:  # 捕获所有数据库相关错误
            logger.error(f"获取 MySQL 模式失败 {self.name}: {err}")
            # 确保在发生错误时，如果连接是临时打开的，它会被关闭
            # 而不是依赖于调用者（如 scan）来处理
            if should_close_conn and conn and conn == self._engine: # 如果是为这个方法特意打开的连接
                conn.dispose()
                self._engine = None # 重置引擎状态
                self._inspector = None # 重置 inspector
            raise RuntimeError(f"获取 MySQL 模式失败: {err}") from err
        finally:
            # cursor 不再直接在此方法级别管理，因为 inspector.get_columns 等方法内部会处理游标
            # if cursor:
            #     cursor.close()
            if should_close_conn and conn:
                if conn == self._engine: # 只有当此方法创建了 self._engine 时才关闭
                    conn.dispose()
                    self._engine = None
                    self._inspector = None # 也清空 inspector
                elif conn != self._engine: # 如果是临时创建的 conn，且不是 self._engine
                    conn.dispose()

    def sampleData(self, conceptName: str, count: int = 10) -> List[Dict[str, Any]]:
        """为指定的 Concept 名称从 MySQL 数据源检索样本数据。

        它通过将 Concept 名称转换回可能的表名（假设遵循 scan 方法中的命名约定），
        然后查询该表来获取指定数量的样本行。

        Args:
            conceptName (str): 要检索样本数据的 Concept 的名称。
            count (int): 要检索的样本行数，默认为 10。

        Returns:
            List[Dict[str, Any]]: 一个字典列表，每个字典代表一行数据，
                                  其中键是列名，值是对应的数据。
                                  如果找不到 Concept 或发生错误，则返回空列表。
        """
        if count <= 0:
            logger.info(f"Sample count is {count}, returning empty list for concept '{conceptName}' in {self.name}.")
            return []

        target_table_name: Optional[str] = None
        actual_column_names: List[str] = []

        try:
            db_schema = self.get_schema() # This might connect if not connected.
            if not db_schema:
                logger.warning(f"Could not retrieve schema for {self.name} to find concept '{conceptName}'.")
                return []

            for table_name_from_schema, columns_details in db_schema.items():
                # Ensure table_name_from_schema is a string for _camelCase
                if not isinstance(table_name_from_schema, str):
                    logger.warning(f"Skipping non-string table name in schema: {table_name_from_schema}")
                    continue

                generated_concept_name = _camelCase(table_name_from_schema).capitalize()
                if generated_concept_name == conceptName:
                    target_table_name = table_name_from_schema
                    actual_column_names = [
                        col_info['name']
                        for col_info in columns_details
                        if col_info.get('name')
                    ]
                    if not actual_column_names:
                        logger.warning(f"Concept '{conceptName}' (Table '{target_table_name}') found in {self.name} but has no columns. Cannot sample data.")
                        return [] # Cannot select data if no columns
                    break # Found the table

            if not target_table_name:
                logger.warning(f"Concept '{conceptName}' not found as a discoverable table in datasource '{self.name}'.")
                return []

            # Quoting column names and table name for the SQL query
            quoted_column_names_str = ", ".join([f"`{col}`" for col in actual_column_names])
            # Ensure target_table_name is just the name, not schema.name, etc.
            # get_schema() returns table names as keys, so this should be fine.
            sql_query = f"SELECT {quoted_column_names_str} FROM `{target_table_name}` LIMIT {count}"

            logger.debug(f"Executing sample data query for concept '{conceptName}' on {self.name}: {sql_query}")
            query_result = self.executeQuery(sql_query) # fetchColumns is True by default

            result_columns = query_result.get('columns', [])
            result_data_rows = query_result.get('data', [])

            if not result_columns and result_data_rows:
                logger.warning(f"Query for concept '{conceptName}' in {self.name} returned data but no column names. This might indicate an issue with executeQuery or the underlying table structure.")
                # Attempt to use actual_column_names if order and count match, but this is risky.
                # Sticking to result_columns from executeQuery is safer.
            
            formatted_samples: List[Dict[str, Any]] = []
            for row_tuple in result_data_rows:
                processed_row = []
                for item in row_tuple:
                    if isinstance(item, (datetime.datetime, datetime.date, datetime.time)):
                        processed_row.append(item.isoformat())
                    else:
                        processed_row.append(item)

                if len(processed_row) == len(result_columns):
                    formatted_samples.append(dict(zip(result_columns, processed_row)))
                else:
                    logger.warning(
                        f"Row data length mismatch for concept '{conceptName}' in {self.name}. "
                        f"Expected {len(result_columns)} columns based on query result, got {len(processed_row)}. Row: {processed_row}"
                    )
            return formatted_samples

        except ConnectionError as ce:
            logger.error(f"Connection error while fetching sample data for concept '{conceptName}' from {self.name}: {ce}")
            return []
        except Exception as e:
            logger.error(f"Error fetching sample data for concept '{conceptName}' (table: {target_table_name or 'unknown'}) from {self.name}: {e}", exc_info=True)
            return []

    # 可以添加 MySQL 特有的方法，例如执行 MySQL 特定的查询
    def execute_mysql_specific_query(self, query: str) -> Dict[str, Any]:
        """执行一个 MySQL 特定的查询并返回结果"""
        if not self._engine:
            self.connect()
        if not self._engine:
            raise ConnectionError(f"无法连接到数据库: {self.name}")

        cursor = None
        try:
            cursor = self._engine.connect().execute(text(query))
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return {"columns": columns, "data": results}
        except Exception as e:
            logger.error(f"执行 MySQL 特定查询时出错 on {self.name}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()