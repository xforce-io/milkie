import pymysql
from typing import Optional, Union
from clients.bird.config import Config, DatabaseConfig
from clients.bird.logger import logger

class Database:
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

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = Config.load().database
        self._config = config  # 保存配置以便重连
        self._connect()
    
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
    
    def __del__(self):
        if hasattr(self, '_db'):
            self._db.close()
