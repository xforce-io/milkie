import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from unittest.mock import Mock, patch
from milkie.agent.llm_block import LLMBlock, Instruction, InstAnalysisResult
from milkie.context import Context
from milkie.response import Response

class TestLLMBlock(unittest.TestCase):

    @patch('milkie.agent.llm_block.Loader.load')
    def setUp(self, mock_load):
        mock_load.return_value = "Test Task"
        self.context = Mock(spec=Context)
        self.context.globalContext.globalConfig.getLLMConfig.return_value.systemPrompt = "Test System Prompt"
        self.context.varDict = {}
        self.llm_block = LLMBlock(context=self.context, task="Test Task")
        self.llm_block.usePrevResult = True

    def test_init(self):
        self.assertEqual(self.llm_block.task, "Test Task")
        self.assertIsNotNone(self.llm_block.taskEngine)
        self.assertFalse(self.llm_block.isCompiled)

    @patch('milkie.agent.llm_block.LLMBlock._decomposeTask')
    def test_compile(self, mock_decompose):
        mock_decompose.return_value = [("1", Mock(spec=Instruction))]
        self.llm_block.compile()
        self.assertTrue(self.llm_block.isCompiled)
        self.assertEqual(len(self.llm_block.instructions), 1)
        self.assertEqual(self.llm_block.instructions[0][0], "1")
        self.assertIsInstance(self.llm_block.instructions[0][1], Mock)

    @patch('milkie.agent.llm_block.TaskEngine.execute')
    def test_execute(self, mock_execute):
        mock_execute.return_value = (True, Response(respStr="Test Response"))
        self.llm_block.isCompiled = True
        result = self.llm_block.execute(query="Test Query", args={"arg1": "value1"})
        self.assertEqual(result.respStr, "Test Response")
        mock_execute.assert_called_once()

    def test_recompile(self):
        self.llm_block.isCompiled = True
        with patch.object(self.llm_block, 'compile') as mock_compile:
            self.llm_block.recompile()
            self.assertFalse(self.llm_block.isCompiled)
            mock_compile.assert_called_once()

    def test_decomposeTask(self):
        task = """
        1. First instruction
        2. Second instruction
        3. Third instruction
        """
        result = self.llm_block._decomposeTask(task)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "1")
        self.assertEqual(result[1][0], "2")
        self.assertEqual(result[2][0], "3")
        self.assertIsInstance(result[0][1], Instruction)
        self.assertIsInstance(result[1][1], Instruction)
        self.assertIsInstance(result[2][1], Instruction)

class TestInstruction(unittest.TestCase):

    def setUp(self):
        self.llm_block = Mock(spec=LLMBlock)
        self.llm_block.getVarDict.return_value = {"var1": "value1"}
        self.llm_block.taskEngine = Mock()
        self.llm_block.taskEngine.instructionRecords = []
        self.llm_block.toolkit = Mock()
        self.llm_block.context = Mock()
        self.llm_block.context.globalContext = Mock()
        self.llm_block.usePrevResult = True

    def test_init(self):
        instruction = Instruction(self.llm_block, "Test Instruction")
        self.assertEqual(instruction.curInstruct, "Test Instruction")
        self.assertIsNone(instruction.label)
        self.assertIsNone(instruction.prev)

    @patch('milkie.agent.llm_block.StepLLMInstAnalysis.run')
    def test_execute(self, mock_run):
        mock_run.return_value = InstAnalysisResult(
            result=InstAnalysisResult.Result.ANSWER,
            funcExecRecords=None,
            response="Test Response"
        )
        instruction = Instruction(self.llm_block, "Test Instruction")
        result = instruction.execute(args={})
        self.assertEqual(result.response.respStr, "Test Response")

if __name__ == '__main__':
    unittest.main()