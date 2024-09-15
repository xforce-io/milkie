import unittest
from unittest.mock import Mock, patch
from milkie.agent.llm_block import LLMBlock, Instruction, AnalysisResult, InstAnalysisResult
from milkie.context import Context
from milkie.response import Response

class TestLLMBlock(unittest.TestCase):

    @patch('milkie.agent.llm_block.Loader.load')
    def setUp(self, mock_load):
        mock_load.return_value = "Test Task"
        self.context = Mock(spec=Context)
        self.context.globalContext.globalConfig.getLLMConfig.return_value.systemPrompt = "Test System Prompt"
        self.llm_block = LLMBlock(context=self.context, task="Test Task")
        
        # 添加这一行来设置 usePrevResult 属性
        self.llm_block.usePrevResult = True

    def test_init(self):
        self.assertEqual(self.llm_block.task, "Test Task")
        self.assertIsNotNone(self.llm_block.tools)
        self.assertIsNotNone(self.llm_block.taskEngine)
        self.assertFalse(self.llm_block.isCompiled)

    @patch('milkie.agent.llm_block.LLMBlock._decomposeTask')
    def test_compile(self, mock_decompose):
        mock_decompose.return_value = AnalysisResult(
            AnalysisResult.Result.DECOMPOSE,
            instructions=[("1", "Test Instruction")]
        )
        self.llm_block.compile()
        self.assertTrue(self.llm_block.isCompiled)
        self.assertEqual(len(self.llm_block.instructions), 1)
        self.assertEqual(self.llm_block.instructions[0][0], "1")
        self.assertIsInstance(self.llm_block.instructions[0][1], Instruction)

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
        result = LLMBlock._decomposeTask(task)
        self.assertEqual(result.result, AnalysisResult.Result.DECOMPOSE)
        self.assertEqual(len(result.instructions), 3)
        self.assertEqual(result.instructions[0], ("1", "First instruction"))
        self.assertEqual(result.instructions[1], ("2", "Second instruction"))
        self.assertEqual(result.instructions[2], ("3", "Third instruction"))

class TestInstruction(unittest.TestCase):

    def setUp(self):
        self.llm_block = Mock(spec=LLMBlock)
        self.llm_block.getVarDict.return_value = {"var1": "value1"}
        self.llm_block.taskEngine = Mock()
        self.llm_block.taskEngine.instructionRecords = []
        self.llm_block.tools = Mock()
        self.llm_block.context = Mock()
        self.llm_block.context.globalContext = Mock()
        
        # 添加这一行来设置 usePrevResult 属性
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