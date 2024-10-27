import logging
import subprocess
from typing import (
    Any,
    Callable,
    TypeVar,
)

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

def mergeDict(dict1 :dict, dict2 :dict):
    result = {**dict1}
    for key, value in dict2.items():
        if key not in result:
            result[key] = value
    return result
