import pymysql
from typing import Optional, Union
from clients.bird.config import Config, DatabaseConfig
from clients.bird.logger import logger

class Database:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = Config.load().database
            
        self._db = pymysql.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database
        )
    
    def execsql(self, sql: str) -> Optional[str]:
        """
        执行SQL查询
        
        Returns:
            str: 非空结果字符串
            "": 空结果字符串
            None: 执行出错
        """
        cursor = None
        try:
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
            
            return str(formatted_results)
            
        except Exception as e:
            logger.error(f"Error executing SQL: {sql}, error: {str(e)}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def __del__(self):
        if hasattr(self, '_db'):
            self._db.close()
