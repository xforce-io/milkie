import json
import os
import logging
import threading
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheKV:
    def __init__(self, filePath: str, dumpInterval: int = 5, expireTimeByDay: float = 1):
        self.filePath = filePath
        self.cache = {}
        self.lock = threading.Lock()
        self.dumpInterval = dumpInterval
        self.lastDumpSec = 0
        self.expireTimeByDay = expireTimeByDay
        self.loadCache()

    def loadCache(self):
        if os.path.exists(self.filePath):
            try:
                with open(self.filePath, 'r', encoding="utf-8") as f:
                    loaded_cache = json.load(f)
                    for key, value in loaded_cache.items():
                        if self._cacheItemExpired(value):
                            continue

                        if isinstance(value, dict) and 'value' in value and 'timestamp' in value:
                            self.cache[key] = value
                        else:
                            # 兼容旧格式
                            self.cache[key] = {'value': value, 'timestamp': time.time()}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading cache file {self.filePath}: {e}")
                raise Exception(f"Error loading cache file {self.filePath}: {e}")

    def dumpCache(self):
        curTime = time.time()
        if curTime - self.lastDumpSec < self.dumpInterval:
            return
        
        with self.lock:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.filePath), exist_ok=True)
            # 清除过期缓存
            self.cache = {key: value for key, value in self.cache.items() if not self._cacheItemExpired(value)}
            with open(self.filePath, 'w', encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
            self.lastDumpSec = curTime

    def _keyToStr(self, key: List[Dict]) -> str:
        return json.dumps(key, sort_keys=True, ensure_ascii=False)

    def get(self, key: List[Dict]) -> Optional[Any]:
        keyStr = self._keyToStr(key)
        with self.lock:
            cached_item = self.cache.get(keyStr)
            if cached_item:
                if not self._cacheItemExpired(cached_item):
                    return cached_item['value']
                else:
                    del self.cache[keyStr]
        return None

    def set(self, key: List[Dict], value: Any):
        keyStr = self._keyToStr(key)
        with self.lock:
            self.cache[keyStr] = {'value': value, 'timestamp': time.time()}
        self.dumpCache()

    def remove(self, key: List[Dict]):
        keyStr = self._keyToStr(key)
        with self.lock:
            if keyStr in self.cache:
                del self.cache[keyStr]
        self.dumpCache()

    def _cacheItemExpired(self, cacheItem: dict) -> bool:
        return time.time() - cacheItem['timestamp'] > self.expireTimeByDay * 86400

class CacheKVMgr:
    FilePrefix = "cache_"
    
    def __init__(self, cacheDir: str, category: str, dumpInterval: int = 5, expireTimeByDay: float = 1):
        self.prefix = f"{CacheKVMgr.FilePrefix}{category}_"
        self.cacheDir = cacheDir
        self.category = category
        self.dumpInterval = dumpInterval
        self.expireTimeByDay = expireTimeByDay
        self.caches = {}
        self.loadCaches()

    def loadCaches(self):
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        
        for fileName in os.listdir(self.cacheDir):
            if fileName.startswith(self.prefix) and fileName.endswith('.json'):
                modelName = fileName[len(self.prefix):-5]
                filePath = os.path.join(self.cacheDir, fileName)
                try:
                    self.caches[modelName] = CacheKV(filePath, self.dumpInterval, self.expireTimeByDay)
                except Exception as e:
                    logger.error(f"Error initializing cache for {modelName}: {e}")
                    raise Exception(f"Error initializing cache for {modelName}: {e}")

    def getCache(self, modelName: str) -> CacheKV:
        return self.caches.get(modelName)

    def getValue(self, modelName: str, key: List[Dict]) -> Optional[Any]:
        cache = self.getCache(modelName)
        if cache:
            return cache.get(key)
        return None

    def setValue(self, modelName: str, key: List[Dict], value: Any):
        cache = self.getCache(modelName)
        if not cache:
            filePath = os.path.join(self.cacheDir, f"{self.prefix}{modelName}.json")
            cache = CacheKV(filePath, self.dumpInterval, self.expireTimeByDay)
            self.caches[modelName] = cache
        cache.set(key, value)

    def removeValue(self, modelName: str, key: List[Dict]):
        cache = self.getCache(modelName)
        if cache:
            cache.remove(key)

class CacheKVCenter:

    def __init__(self):
        self.repos = {}

    def getCacheMgr(
            self, 
            cacheDir: str, 
            category: str, 
            dumpInterval: int = 5, 
            expireTimeByDay: float = 1) -> CacheKVMgr:
        key = f"{cacheDir}_{category}"
        if key not in self.repos:
            self.repos[key] = CacheKVMgr(cacheDir, category, dumpInterval, expireTimeByDay)
        return self.repos[key]

GlobalCacheKVCenter = CacheKVCenter()

if __name__ == "__main__":
    cacheMgr = CacheKVMgr('data/cache', category='test', dumpInterval=10)

    modelName = "exampleModel"
    key = [
        {"role": "系统", "content": "You are an assistant."}, 
        {"role": "用户", "content": "system: \n Role:今天天气如何? \n . "}]
    value = {"data": "some data"}
    cacheMgr.setValue(modelName, key, value)
    
    cacheMgr = CacheKVMgr('data/cache', category='test', dumpInterval=10)
    cachedValue = cacheMgr.getValue(modelName, key)
    assert cachedValue["data"] == "some data"

    cachedValue = cacheMgr.getValue(modelName, {"id": 2})
    assert cachedValue == None 