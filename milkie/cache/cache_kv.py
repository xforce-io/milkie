import json
import os
import logging
import threading
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

class CacheKV:
    def __init__(self, filePath: str, dumpInterval: int = 5):
        self.filePath = filePath
        self.cache = {}
        self.lock = threading.Lock()
        self.dumpInterval = dumpInterval
        self.lastDumpSec = 0
        self.loadCache()

    def loadCache(self):
        if os.path.exists(self.filePath):
            try:
                with open(self.filePath, 'r', encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.log(f"Error loading cache file {self.filePath}: {e}")
                self.cache = {}

    def dumpCache(self):
        curTime = time.time()
        if curTime - self.lastDumpSec < self.dumpInterval:
            return
        
        with self.lock:
            with open(self.filePath, 'w', encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)

    def _keyToStr(self, key: List[Dict]) -> str:
        return json.dumps(key, sort_keys=True, ensure_ascii=False)

    def get(self, key: List[Dict]) -> Dict:
        keyStr = self._keyToStr(key)
        result = None
        with self.lock:
            result = self.cache.get(keyStr)
            if result:
                result = result.copy()
        return result

    def set(self, key: List[Dict], value: Dict):
        keyStr = self._keyToStr(key)
        with self.lock:
            self.cache[keyStr] = value
        self.dumpCache()

class CacheKVMgr:

    FilePrefix = "cache_"
    
    def __init__(self, cacheDir: str, dumpInterval: int = 60):
        self.cacheDir = cacheDir
        self.dumpInterval = dumpInterval
        self.caches = {}
        self.loadCaches()

    def loadCaches(self):
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        
        for fileName in os.listdir(self.cacheDir):
            if fileName.startswith(CacheKVMgr.FilePrefix) and fileName.endswith('.json'):
                modelName = fileName[len(CacheKVMgr.FilePrefix):-5]
                filePath = os.path.join(self.cacheDir, fileName)
                try:
                    self.caches[modelName] = CacheKV(filePath, self.dumpInterval)
                except Exception as e:
                    logger.error(f"Error initializing cache for {modelName}: {e}")

    def getCache(self, modelName: str) -> CacheKV:
        return self.caches.get(modelName)

    def getValue(self, modelName: str, key: List[Dict]) -> Dict:
        cache = self.getCache(modelName)
        if cache:
            return cache.get(key)
        return None

    def setValue(self, modelName: str, key: List[Dict], value: Dict):
        cache = self.getCache(modelName)
        if not cache:
            filePath = os.path.join(self.cacheDir, f"cache_{modelName}.json")
            cache = CacheKV(filePath, self.dumpInterval)
            self.caches[modelName] = cache
        cache.set(key, value)

if __name__ == "__main__":
    cacheMgr = CacheKVMgr('data/cache', dumpInterval=10)

    modelName = "exampleModel"
    key = [
        {"role": "系统", "content": "You are an assistant."}, 
        {"role": "用户", "content": "system: \n Role:今天天气如何? \n . "}]
    value = {"data": "some data"}
    cacheMgr.setValue(modelName, key, value)
    
    cacheMgr = CacheKVMgr('data/cache', dumpInterval=10)
    cachedValue = cacheMgr.getValue(modelName, key)
    assert cachedValue["data"] == "some data"

    cachedValue = cacheMgr.getValue(modelName, {"id": 2})
    assert cachedValue == None 