import json
import os
import logging
import threading
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, filePath: str, dumpInterval: int = 60):
        self.filePath = filePath
        self.cache = {}
        self.lock = threading.Lock()
        self.dumpInterval = dumpInterval
        self.stopEvent = threading.Event()
        self.loadCache()
        self.startDumpThread()

    def loadCache(self):
        if os.path.exists(self.filePath):
            try:
                with open(self.filePath, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.log(f"Error loading cache file {self.filePath}: {e}")
                self.cache = {}

    def dumpCache(self):
        with self.lock:
            with open(self.filePath, 'w') as f:
                json.dump(self.cache, f)

    def startDumpThread(self):
        def dumpPeriodically():
            while not self.stopEvent.is_set():
                time.sleep(self.dumpInterval)
                if not self.stopEvent.is_set():
                    self.dumpCache()

        self.dumpThread = threading.Thread(target=dumpPeriodically, daemon=True)
        self.dumpThread.start()

    def stopDumpThread(self):
        self.stopEvent.set()
        self.dumpThread.join()

    def _keyToStr(self, key: List[Dict]) -> str:
        return json.dumps(key, sort_keys=True)

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

class CacheMgr:

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
            if fileName.startswith(CacheMgr.FilePrefix) and fileName.endswith('.json'):
                modelName = fileName[len(CacheMgr.FilePrefix):-5]  # 去掉 'cache_' 和 '.json'
                filePath = os.path.join(self.cacheDir, fileName)
                try:
                    self.caches[modelName] = Cache(filePath, self.dumpInterval)
                except Exception as e:
                    logger.error(f"Error initializing cache for {modelName}: {e}")

    def getCache(self, modelName: str) -> Cache:
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
            cache = Cache(filePath, self.dumpInterval)
            self.caches[modelName] = cache
        cache.set(key, value)

    def stopAllCaches(self):
        for cache in self.caches.values():
            cache.stopDumpThread()

    def __del__(self):
        self.stopAllCaches()

# 示例用法
if __name__ == "__main__":
    cacheMgr = CacheMgr('data/cache', dumpInterval=10)

    modelName = "exampleModel"
    key = [{"id": 1}, {"name": "example"}]
    value = {"data": "some data"}
    cacheMgr.setValue(modelName, key, value)

    cachedValue = cacheMgr.getValue(modelName, key)
    assert cachedValue["data"] == "some data"

    cachedValue = cacheMgr.getValue(modelName, {"id": 2})
    assert cachedValue == None 