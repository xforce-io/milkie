import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "bird") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "bird.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 创建全局日志对象
logger = setup_logger() 

MaxLogLen = 4000

def INFO(msg: str):
    logger.info(msg[:MaxLogLen])

def WARN(msg: str):
    logger.warning(msg[:MaxLogLen])

def ERROR(msg: str):
    logger.error(msg[:MaxLogLen])