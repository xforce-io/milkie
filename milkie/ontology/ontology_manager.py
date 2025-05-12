import threading
import time
import logging
from typing import Optional

from milkie.config.config import GlobalConfig

from .ontology import Ontology

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    管理 Ontology 实例并负责定期同步数据源。
    """
    def __init__(
            self, 
            globalConfig: GlobalConfig,
            syncIntervalSeconds: int = 60):
        """
        初始化 OntologyManager。

        Args:
            syncIntervalSeconds (int): 数据源同步的时间间隔（秒），默认为 1 小时。
        """
        self.ontology: Ontology = Ontology(globalConfig)
        self.syncIntervalSeconds: int = syncIntervalSeconds
        self._syncThread: Optional[threading.Thread] = None
        self._stopEvent: threading.Event = threading.Event()
        self._syncLock: threading.Lock = threading.Lock()
        logger.info(f"OntologyManager 初始化完成，同步间隔: {syncIntervalSeconds} 秒")

        self.ontology.buildOntologyFromSources(runScan=True, concurrent=True)
        logger.info("OntologyManager 本体构建完成")

    def _synchronizeDataSource(self) -> None:
        """
        后台同步任务，定期触发 Ontology 的数据源扫描和构建。
        """
        logger.info("本体同步线程已启动")
        while not self._stopEvent.is_set():
            try:
                logger.info("开始执行本体同步...")
                with self._syncLock:
                    # 实际的同步逻辑，这里调用 Ontology 的方法
                    # 假设 Ontology 有一个 buildOntologyFromSources 方法用于同步
                    # 这里的参数可能需要根据实际情况调整
                    self.ontology.buildOntologyFromSources(runScan=True, concurrent=True)
                logger.info("本体同步完成")

            except Exception as e:
                logger.exception(f"本体同步过程中发生错误: {e}")

            # 等待下一个同步周期或停止事件
            self._stopEvent.wait(self.syncIntervalSeconds)

        logger.info("本体同步线程已停止")

    def start(self) -> None:
        """
        启动后台同步线程。
        """
        if self._syncThread is not None and self._syncThread.is_alive():
            logger.warning("同步线程已在运行中")
            return

        self._stopEvent.clear()
        self._syncThread = threading.Thread(target=self._synchronizeDataSource, daemon=True)
        self._syncThread.start()
        logger.info("已请求启动本体同步线程")

    def stop(self) -> None:
        """
        停止后台同步线程。
        """
        if self._syncThread is None or not self._syncThread.is_alive():
            logger.info("同步线程未运行")
            return

        logger.info("正在请求停止本体同步线程...")
        self._stopEvent.set()
        # 可以选择等待线程结束，或立即返回
        # self._syncThread.join()
        # logger.info("同步线程已确认停止")

    def triggerSyncNow(self) -> None:
        """
        立即手动触发一次本体数据源同步。
        """
        logger.info("开始执行手动本体同步...")
        try:
            with self._syncLock:
                # 假设 Ontology 有一个 buildOntologyFromSources 方法用于同步
                # 这里的参数可能需要根据实际情况调整
                self.ontology.buildOntologyFromSources(runScan=True, concurrent=True)
            logger.info("手动本体同步完成")
        except Exception as e:
            logger.exception(f"手动本体同步过程中发生错误: {e}")

    def getOntology(self) -> Ontology:
        return self.ontology 
    
    def getConcepts(self, concepts: list) -> list:
        return self.ontology.getConcepts(concepts)

    def getDataSourcesFromConcepts(self, concepts: list) -> list:
        return self.ontology.getDataSourcesFromConcepts(concepts)

    def getDataSourceSchemasFromConcepts(self, concepts: list) -> list:
        return self.ontology.getDataSourceSchemasFromConcepts(concepts)
