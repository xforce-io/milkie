import os, sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from milkie.agent.llm_block.llm_block import LLMBlock, Instruction, InstAnalysisResult
from milkie.context import Context
from milkie.response import Response

class TestLLMBlock(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        configPath = os.path.join(os.path.dirname(__file__), '../../config/global_test.yaml')
        self.context = Context.create(configPath)
        
        # 创建 LLMBlock 实例
        self.llmBlock = LLMBlock(
            context=self.context,
            taskExpr="Test Task"
        )

    def testInit(self):
        """测试初始化"""
        llmBlock = LLMBlock(context=self.context, taskExpr="Test Task")
        self.assertEqual(llmBlock.task, "Test Task")
        self.assertTrue(hasattr(llmBlock, 'taskEngine'))

    def testCompile(self):
        """测试编译功能"""
        self.llmBlock.compile()
        self.assertTrue(self.llmBlock.isCompiled)
        self.assertTrue(len(self.llmBlock.instructions) > 0)

    def testDecomposeTask(self):
        """测试任务分解"""
        task = """
        1. First instruction
        2. Second instruction
        3. Third instruction
        """
        instructions = self.llmBlock._decomposeTask(task)
        self.assertEqual(len(instructions), 3)
        self.assertEqual(instructions[0][0], "1")
        self.assertEqual(instructions[1][0], "2")
        self.assertEqual(instructions[2][0], "3")

    def testRecompile(self):
        """测试重新编译"""
        self.llmBlock.compile()
        original_instructions = self.llmBlock.instructions.copy()
        
        self.llmBlock.recompile()
        self.assertEqual(len(self.llmBlock.instructions), len(original_instructions))

class TestInstruction(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        configPath = os.path.join(os.path.dirname(__file__), '../../config/global_test.yaml')
        self.context = Context.create(configPath)
        
        # 创建 LLMBlock 实例
        self.llmBlock = LLMBlock(
            context=self.context,
            taskExpr="Test Task"
        )

    def testInit(self):
        """测试指令初始化"""
        instruction = Instruction(self.llmBlock, "Test Instruction")
        self.assertEqual(instruction.curInstruct, "Test Instruction")
        self.assertIsNone(instruction.label)
        self.assertIsNone(instruction.prev)

if __name__ == '__main__':
    unittest.main()