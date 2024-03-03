import time, logging
from milkie.agent.team.mrqa import MapReduceQA
from milkie.config.config import GlobalConfig
from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class TestCase(object):
    def __init__(self, query :str, answers :list) -> None:
         self.query = query
         self.answers = answers

    def check(self, result) -> bool:
        for answer in self.answers:
            if type(answer) == str:
                if answer not in result:
                    return False
            elif type(answer) == list:
                isRight = False
                for item in answer:
                    if item in result:
                        isRight = True
                        break

                if not isRight:
                    return False
        return True

class TestSuite(object):
    def __init__(self, name, testCases) -> None:
        self.name = name
        self.testCases = testCases

    def run(
            self, 
            ex, 
            globalConfig :GlobalConfig,
            modelFactory :ModelFactory):
        globalContext = GlobalContext(globalConfig, modelFactory)
        context = Context(globalContext=globalContext)
        agent = MapReduceQA(context, "retrieval")

        cnt = 0
        totalTime = 0
        for testCase in self.testCases:
            t0 = time.time()
            result = agent.task(testCase.query)
            t1 = time.time()
            if testCase.check(result.response.message.content):
                cnt += 1
                logger.info(f"Testcase[{testCase.query}] passed cost[{t1-t0:.2f}]]")
            else:
                logger.info(f"Testcase[{testCase.query}] failed Expected[{testCase.answers}] Actual[{result}] cost[{t1-t0:.2f}]]")
            totalTime += t1-t0
        logger.info(f"Running testsuite[{self.name}] "
                    f"accuracy[{cnt}/{len(self.testCases)}] "
                    f"cost[{totalTime}] "
                    f"avg[{totalTime/len(self.testCases)}] ")
        ex.log_scalar("succ", cnt)
        ex.log_scalar("total", len(self.testCases))
        ex.log_scalar("accuracy", cnt/len(self.testCases))
        ex.log_scalar("costMs", totalTime)

