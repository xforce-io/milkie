from abc import abstractmethod
import logging
import json

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
    
    def __init__(self, config) -> None:
        self.isOr = False
        if '$and' in config:
            self.subFilters = [KeywordFilter(cond) for cond in config['$and']]
        elif '$or' in config:
            self.subFilters = [KeywordFilter(cond) for cond in config['$or']]
            self.isOr = True
        elif isinstance(config, str):
            self.subFilters = [KeywordFilter(config)]
        else:
            raise ValueError("Unknown filter format")
    
    def match(self, text) -> bool:
        if self.isOr:
            return any(subFilter.match(text) for subFilter in self.subFilters)
        else:
            return all(subFilter.match(text) for subFilter in self.subFilters)

class TestcaseKeyword:
    
    def __init__(self, config) -> None:
        self.input = config["input"]
        self.context = config["context"]
        self.filter = KeywordFilter(config["keypoints"])
    
    def eval(self, resp) -> bool:
        return self.filter.match(resp)

class BenchType(object):

    @abstractmethod
    def eval(self, text) -> float:
        pass

class BenchTypeKeyword(BenchType):

    def __init__(self, filepathTest :str) -> None:
        self.filepathTest = filepathTest
        self.succ = 0
        self.fail = 0
        
        testcases = []
        with open(filepathTest, 'r') as file:
            for line in file:
                jsonObj = json.loads(line)
                testcase = TestcaseKeyword(jsonObj)
                testcases.append(testcase)
    
    def eval(self, pairsTestcaseAndResp :list):
        for testcase, resp in pairsTestcaseAndResp:
            if testcase.eval(resp):
                self.succ += 1
            else:
                self.fail += 1

    def getAccuracy(self) -> float:
        return float(self.succ) / (self.succ + self.fail)

class Benchmarks(object):
    
    def __init__(self, benchmarks :list) -> None:
        self.benchmarks = benchmarks

    def eval(self, pairsTestcaseAndResp :list):
        for benchmark in self.benchmarks:
            benchmark.eval(pairsTestcaseAndResp)

    def report(self):
        for benchmark in self.benchmarks:
            logger.info(f"benchmark[{benchmark.filepathTest}] succ[{benchmark.succ}] fail[{benchmark.fail}] accuracy[{benchmark.getAccuracy()}]")
    
    def evalAndReport(self, pairsTestcaseAndResp :list):
        self.eval(pairsTestcaseAndResp)
        self.report()