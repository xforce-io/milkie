import unittest
import os
import sys

# 将项目根目录添加到 Python 路径
projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests/unittest'))
sys.path.insert(0, projectRoot)

def runAllTests():
    # 发现所有测试
    testLoader = unittest.TestLoader()
    testSuite = testLoader.discover(start_dir=projectRoot, pattern='test_*.py')

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(testSuite)

    # 返回测试结果
    return result.wasSuccessful()

if __name__ == '__main__':
    success = runAllTests()
    sys.exit(0 if success else 1)
