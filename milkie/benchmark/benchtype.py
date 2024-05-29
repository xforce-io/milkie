from abc import abstractmethod
import logging
import json
import threading
import time
from typing import Callable
from sacred import Experiment
from llama_index.core import Response
from llama_index.core.utils import truncate_text

logger = logging.getLogger(__name__)

def jsonToFilter(config, text):
    if '$and' in config:
        return all(jsonToFilter(cond, text) for cond in config['$and'])
    
    if '$or' in config:
        return any(jsonToFilter(cond, text) for cond in config['$or'])
    
    if isinstance(config, str):
        return config in text

    raise ValueError("Unknown filter format")

class KeywordFilter:

    RefuseKeywords = ["无法得出", "无法知道", "不知道", "不确定", "未找到", "抱歉","无法确定","没有提到","无法回答","没有明确回答","没有找到","无法找到","未找到","未提及","unknown","无法判断","无关","无法直接回答","没有明确"]
    
    def __init__(self, config) -> None:
        self.isOr = None
        if '$and' in config:
            self.filters = [KeywordFilter(cond) for cond in config['$and']]
            self.isOr = False
        elif '$or' in config:
            self.filters = [KeywordFilter(cond) for cond in config['$or']]
            self.isOr = True
        elif isinstance(config, str):
            self.filters = config
        elif isinstance(config, dict):
            self.filters = None
        else:
            raise ValueError("Unknown filter format")
    
    def match(self, text) -> bool:
        if self.isOr == None:
            if self.filters == None:
                return any(keyword in text for keyword in KeywordFilter.RefuseKeywords)
            else:
                return self.filters in text
        elif self.isOr:
            return any(subFilter.match(text) for subFilter in self.filters)
        else:
            return all(subFilter.match(text) for subFilter in self.filters)

class TestcaseKeyword:
    
    def __init__(self, config) -> None:
        self.input = config["input"]
        self.context = config["context"] if "context" in config else None
        self.filter = KeywordFilter(json.loads(config["keypoints"]))
        self.keypoints = config["keypoints"]

    def eval(self, resp) -> bool:
        return self.filter.match(resp.response)

class BenchType(object):

    def __init__(self, filepathTest :str) -> None:
        self.filepathTest = filepathTest
        self.name = filepathTest.split("/")[-1].split(".")[0]

    @abstractmethod
    def eval(
            self, 
            agent: Callable[[str, dict], list[Response]],
            prompt :str,
            batchSize :int) -> float:
        pass

    @abstractmethod
    def evalParrel(self,
            agent: Callable[[str, dict], list[Response]],
            prompt :str,
            batchSize :int) -> float:
        pass

class BenchTypeKeyword(BenchType):

    class Statistics :

        def __init__(self) -> None:
            self.succ = 0
            self.fail = 0
            self.lockStatics = threading.Lock()
            self.startMs = 0
            self.stopMs = 0

        def tick(self):
            self.startMs = time.time()
        
        def tock(self):
            self.stopMs = time.time()

        def addSucc(self):
            with self.lockStatics:
                self.succ += 1
                
        def addFail(self):
            with self.lockStatics:
                self.fail += 1

        def costMs(self) -> float:
            return self.stopMs - self.startMs

    statistics = Statistics()

    def __init__(self, filepathTest :str) -> None:
        super().__init__(filepathTest)
        
        self.testcases = []
        with open(filepathTest, 'r') as file:
            for line in file:
                jsonObj = json.loads(line)
                try:
                    testcase = TestcaseKeyword(jsonObj)
                    self.testcases.append(testcase)
                except Exception as e:
                    logger.error(f"Error[{e}] in parsing line[{line}]")
    
    def eval(
            self, 
            agent: Callable[[str, dict], list[Response]], 
            prompt :str,
            batchSize :int):
        for i in range(0, len(self.testcases), batchSize):
            batch = self.testcases[i:i+batchSize]
            argsList = [{"query_str": testcase.input, "context_str": testcase.context} for testcase in batch]
            responses = agent(prompt=prompt, argsList=argsList)
            for j, response in enumerate(responses):
                status = f'Testcase[{batch[j].input[:30]}] Ans[{truncate_text(response.response, 500)}] Keypoints[{batch[j].keypoints}]'.replace("\n", "//")
                if batch[j].eval(response):
                    self.succ += 1
                    logger.info(f"{status} succ")
                else:
                    self.fail += 1
                    logger.info(f"{status} fail")

    def evalParrel(
            self,
            agent: Callable[[str, dict], list[Response]],
            prompt :str,
            batchSize :int):
        self.statistics.tick()
        
        threads = []
        for i in range(0, len(self.testcases), batchSize):
            batch = self.testcases[i:i+batchSize]
            t = threading.Thread(
                target=BenchTypeKeyword._callAgent,
                args=(agent, prompt, batch, self.statistics),
                daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        self.statistics.tock()

    @property
    def succ(self) -> int: 
        return self.statistics.succ

    @property
    def fail(self) -> int:
        return self.statistics.fail

    @property
    def costMs(self) -> float:
        return self.statistics.costMs()

    def getAccuracy(self) -> float:
        return float(self.succ) / (self.succ + self.fail)

    def _callAgent(
            agent: Callable[[str, dict], list[Response]],
            prompt :str,
            batch: list[TestcaseKeyword],
            statistics: Statistics):
        argsList = [{"query_str": testcase.input, "context_str": testcase.context} for testcase in batch]
        responses = agent(prompt=prompt, argsList=argsList)
        for j, response in enumerate(responses):
            status = f'Testcase[{batch[j].input[:30]}] Ans[{truncate_text(response.response, 500)}] Keypoints[{batch[j].keypoints}]'.replace("\n", "//")
            if batch[j].eval(response):
                statistics.addSucc()
                logger.info(f"{status} succ")
            else:
                statistics.addFail()
                logger.info(f"{status} fail")

class Benchmarks(object):
    
    def __init__(self, ex :Experiment, benchmarks :list[BenchType], batchSize :int) -> None:
        self.ex = ex
        self.benchmarks = benchmarks
        self.batchSize = batchSize

    def eval(
            self, 
            agent: Callable[[str, dict], list[Response]], 
            prompt :str):
        for benchmark in self.benchmarks:
            benchmark.evalParrel(agent, prompt, self.batchSize)

    def report(self):
        for benchmark in self.benchmarks:
            self.ex.log_scalar(f"benchmark.{benchmark.name}.succ", benchmark.succ)
            self.ex.log_scalar(f"benchmark.{benchmark.name}.fail", benchmark.fail)
            self.ex.log_scalar(f"benchmark.{benchmark.name}.accu", benchmark.getAccuracy())

            logger.info(
                f"Benchmark[{benchmark.name}] "
                f"succ[{benchmark.succ}] "
                f"fail[{benchmark.fail}] "
                f"accu[{benchmark.getAccuracy()}] "
                f"cost[{benchmark.costMs}] "
                f"avg[{benchmark.costMs / (benchmark.succ + benchmark.fail)}] ")
    
    def evalAndReport(
            self, 
            agent: Callable[[str, dict], list[Response]], 
            prompt :str):
        self.eval(agent=agent, prompt=prompt)
        self.report()