import os, sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from milkie.agent.flow_block import FlowBlock
from milkie.agent.for_block import ForBlock
from milkie.agent.llm_block.llm_block import LLMBlock, Response
from milkie.context import Context, VarDict

class MockLLMBlock(LLMBlock):
    """模拟 LLMBlock 的行为"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isCompileCalled = False
        self.isExecuteCalled = False
        self.executeArgs = {}

    def compile(self):
        self.isCompileCalled = True

    def execute(self, context: Context, query=None, args={}, prevBlock=None):
        self.isExecuteCalled = True
        self.executeArgs = args.copy() if isinstance(args, dict) else args.getAllDict()
        return Response(respStr="Mock LLM response", metadata={"args": self.executeArgs})

class TestFlowBlock(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        config_path = os.path.join(os.path.dirname(__file__), '../../config/global_test.yaml')
        self.context = Context.create(config_path)

    def testBasicCompile(self):
        """测试基本的编译功能"""
        flowCode = "1. Simple task"
        flowBlock = FlowBlock(
            flowCode,
            context=self.context,
        )
        flowBlock.compile()
        
        self.assertEqual(len(flowBlock.blocks), 1)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)

    def testCompileWithForLoop(self):
        """测试包含 for 循环的编译"""
        flowCode = """
        1. First task
        FOR item in items
            2. Process {item}
        END
        3. Final task
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        
        self.assertEqual(len(flowBlock.blocks), 3)
        self.assertIsInstance(flowBlock.blocks[0], LLMBlock)
        self.assertIsInstance(flowBlock.blocks[1], ForBlock)
        self.assertIsInstance(flowBlock.blocks[2], LLMBlock)

    def testCompileWithReturnValue(self):
        """测试带返回值的 for 循环编译"""
        flowCode = """
        1. First task
        FOR item in items
            2. Process {item}
        END -> result
        3. Use {result}
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        
        self.assertEqual(len(flowBlock.blocks), 3)
        self.assertEqual(flowBlock.blocks[1].retStorage, "result")

    @patch('milkie.agent.llm_block.llm_block.LLMBlock', MockLLMBlock)
    def testExecute(self):
        """测试执行流程"""
        flowCode = """
        1. First task
        2. Second task
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        
        result = flowBlock.execute(context=self.context)
        self.assertIsInstance(result, Response)
        self.assertEqual(result.respStr, "Mock LLM response")

    @patch('milkie.agent.llm_block.llm_block.LLMBlock', MockLLMBlock)
    def testExecuteWithForLoop(self):
        """测试带 for 循环的执行"""
        flowCode = """
        1. First task
        FOR item in items
            2. Process {item}
        END
        3. Final task
        """
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        
        varDict = VarDict()
        varDict.setGlobal("items", [1, 2, 3])
        result = flowBlock.execute(context=self.context, args=varDict.getAllDict())
        
        self.assertIsInstance(result, Response)
        self.assertEqual(result.respStr, "Mock LLM response")

    def testInvalidForLoop(self):
        """测试无效的 for 循环语法"""
        invalid_codes = [
            # 无效的变量名
            """
            FOR item in items
                Process item
            END -> 123invalid
            """,
            # 错误的 END 语法
            """
            FOR item in items
                Process item
            END ->> result
            """
        ]
        
        for code in invalid_codes:
            with self.subTest(code=code):
                flowBlock = FlowBlock(code, context=self.context)
                with self.assertRaises((SyntaxError, ValueError)):
                    flowBlock.compile()

    def testRecompile(self):
        """测试重复编译"""
        flowCode = "1. Simple task"
        flowBlock = FlowBlock(flowCode, context=self.context)
        flowBlock.compile()
        original_blocks = flowBlock.blocks.copy()
        
        # 重复编译不应改变结果
        flowBlock.compile()
        self.assertEqual(len(flowBlock.blocks), len(original_blocks))
        self.assertEqual(flowBlock.blocks, original_blocks)

if __name__ == '__main__':
    unittest.main()