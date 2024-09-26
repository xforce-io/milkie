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

    def testSingleForInstruction(self):
        flowCode = "for the next step, do something"
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        self.assertEqual(len(flowBlock.blocks), 1)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)

    def testEndInstructionOutsideForLoop(self):
        flowCode = """
        1. First step
        2. Second step
        3. end the process here
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        self.assertEqual(len(flowBlock.blocks), 1)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)

    def testIncompleteForLoop(self):
        flowCode = """
        1. First step
        FOR item in items:
            2. Process {item}
        3. This should be part of the for loop but there's no 'end'
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        self.assertEqual(len(flowBlock.blocks), 2)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)
        self.assertIsInstance(flowBlock.blocks[1], ForBlock)

    def testMultipleForLoops(self):
        flowCode = """
        1. First step
        2. Second step
        FOR item in items:
            3. Process {item}
        END
        4. Middle step
        FOR x in range:
            5. Do {x}
        END
        6. Later step
        7. Final step
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        self.assertEqual(len(flowBlock.blocks), 5)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)
        self.assertIsInstance(flowBlock.blocks[1], ForBlock)
        self.assertIsInstance(flowBlock.blocks[2], LLMBlock)
        self.assertIsInstance(flowBlock.blocks[3], ForBlock)
        self.assertIsInstance(flowBlock.blocks[4], LLMBlock)

if __name__ == '__main__':
    unittest.main()