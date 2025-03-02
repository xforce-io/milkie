import logging
import subprocess
from typing import (
    Any,
    Callable,
    List,
    TypeVar,
)

from milkie.functions.openai_function import OpenAIFunction

F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def getMemStat():
    logger.info("================Getting GPU memory status====================")
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    lines = output.strip().split('\n')
    for line in lines:
        total, used, free = line.split(',')
        logger.info(f"Total memory: {total} MB, Used memory: {used} MB, Free memory: {free} MB")
    logger.info("============================================================")

def addDict(dict1 :dict, dict2 :dict):
    result = {**dict1}
    for key, value in dict2.items():
        result[key] = value
    return result

def getToolsSchemaForTools(tools: List[OpenAIFunction]) -> list:
    return [tool.get_openai_tool_schema() for tool in tools]