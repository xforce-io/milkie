import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from milkie.agent.flow_block import FlowBlock
from milkie.agent.llm_block import LLMBlock
from milkie.agent.for_block import ForBlock
from milkie.context import Context

class TestFlowBlock(unittest.TestCase):
    def setUp(self):
        self.context = Context.create("config/global.yaml")
        self.flowCode = """
        1. First step
        2. Second step
        FOR item in items:
            3. Process {item}
        END 
        4. Final step
        """
        self.flowBlock = FlowBlock(self.flowCode, context=self.context)

    def testCompile(self):
        self.flowBlock.compile()
        self.assertEqual(len(self.flowBlock.blocks), 3)
        self.assertIsInstance(self.flowBlock.blocks[0], LLMBlock)
        self.assertIsInstance(self.flowBlock.blocks[1], ForBlock)
        self.assertIsInstance(self.flowBlock.blocks[2], LLMBlock)

    # ... 其他测试方法保持不变 ...

    # 移除与 FuncBlock 相关的测试方法

if __name__ == '__main__':
    unittest.main()