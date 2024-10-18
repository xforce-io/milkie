import os, sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from milkie.agent.agent import Agent
from milkie.agent.flow_block import FlowBlock
from milkie.agent.func_block import FuncBlock
from milkie.context import Context
from milkie.config.constant import KeywordFuncStart, KeywordFuncEnd
from milkie.functions.toolkits.toolkit import Toolkit

logging.basicConfig(level=logging.DEBUG)

class MockToolkit(Toolkit):
    def getToolsSchema(self):
        return []

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.context = Context.create("config/global.yaml")
        self.toolkit = MockToolkit()

    def testCompile(self):
        code = f"""
        1. First step
        {KeywordFuncStart} myFunc(param1, param2):
            2. Do something with {{param1}}
            3. Do something with {{param2}}
        {KeywordFuncEnd}
        4. Second step
        5. Third step
        {KeywordFuncStart} anotherFunc(x):
            6. Do something with {{x}}
        {KeywordFuncEnd}
        7. Final step
        """
        agent = Agent(code, context=self.context, toolkit=self.toolkit)
        agent.compile()

        self.assertEqual(len(agent.repoFuncs.funcs), 2)
        self.assertEqual(len(agent.flowBlocks), 3)
        self.assertIsInstance(agent.repoFuncs.get("myFunc"), FuncBlock)
        self.assertIsInstance(agent.repoFuncs.get("anotherFunc"), FuncBlock)
        self.assertIsInstance(agent.flowBlocks[0], FlowBlock)
        self.assertIsInstance(agent.flowBlocks[1], FlowBlock)
        self.assertIsInstance(agent.flowBlocks[2], FlowBlock)

if __name__ == '__main__':
    unittest.main()