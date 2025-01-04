import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
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
        return Response(respStr="Mock response", metadata={"args": self.executeArgs})

class TestForBlock(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        configPath = os.path.join(os.path.dirname(__file__), '../../config/global_test.yaml')
        self.context = Context.create(configPath)
        self.basicForStatement = "FOR item in items\n    1. Process {item}\nEND"

    def createForBlock(self, forStatement=None, loopBlockClass=None):
        """创建 ForBlock 实例的辅助方法"""
        return ForBlock(
            context=self.context,
            forStatement=forStatement or self.basicForStatement,
            loopBlockClass=loopBlockClass or MockLLMBlock
        )

    def createVarDict(self, items):
        """创建 VarDict 实例的辅助方法"""
        varDict = VarDict()
        varDict.setGlobal("items", items)
        return varDict

    def testParseForStatement(self):
        """测试 FOR 语句的解析"""
        forBlock = self.createForBlock()
        self.assertEqual(forBlock.loopVar, "item")
        self.assertEqual(forBlock.iterable, "items")
        self.assertEqual(forBlock.loopBody, "1. Process {item}\nEND")

    def testInvalidForStatement(self):
        """测试无效的 FOR 语句"""
        invalidStatements = [
            "invalid for statement",
            "FOR in items",
            "FOR item items",
            "FOR item in",
        ]
        for stmt in invalidStatements:
            with self.subTest(statement=stmt):
                with self.assertRaises(ValueError):
                    self.createForBlock(stmt)

    def testEmptyLoopBody(self):
        """测试空循环体"""
        forBlock = self.createForBlock("FOR item in items")
        self.assertEqual(forBlock.loopBody, "pass")

    def testCompile(self):
        """测试编译过程"""
        forBlock = self.createForBlock()
        forBlock.compile()
        self.assertTrue(forBlock.loopBlock.isCompileCalled)
        self.assertIsInstance(forBlock.loopBlock, MockLLMBlock)

    def testValidateList(self):
        """测试列表类型的验证"""
        testCases = [
            [1, 2, 3],
            [],
            ["a", "b", "c"],
            [{"key": "value"}],
        ]
        forBlock = self.createForBlock()
        for items in testCases:
            with self.subTest(items=items):
                varDict = self.createVarDict(items)
                result = forBlock.validate(varDict)
                self.assertEqual(forBlock.loopType, list)
                self.assertEqual(result, items)

    def testValidateDict(self):
        """测试字典类型的验证"""
        testCases = [
            {"a": 1, "b": 2},
            {},
            {"key": "value"},
            {"nested": {"key": "value"}},
        ]
        forBlock = self.createForBlock()
        for items in testCases:
            with self.subTest(items=items):
                varDict = self.createVarDict(items)
                result = forBlock.validate(varDict)
                self.assertEqual(forBlock.loopType, dict)
                self.assertEqual(result, items)

    def testValidateInvalidType(self):
        """测试无效类型的验证"""
        invalidItems = [
            42,
            "string",
            None,
            True,
        ]
        forBlock = self.createForBlock()
        for items in invalidItems:
            with self.subTest(items=items):
                varDict = self.createVarDict(items)
                with self.assertRaises(ValueError):
                    forBlock.validate(varDict)

    def testExecuteList(self):
        """测试列表类型的执行"""
        forBlock = self.createForBlock()
        forBlock.compile()
        varDict = self.createVarDict([1, 2, 3])
        result = forBlock.execute(context=self.context, args=varDict.getAllDict())
        
        self.assertTrue(forBlock.loopBlock.isExecuteCalled)
        self.assertIsInstance(result, Response)
        self.assertEqual(forBlock.loopBlock.executeArgs.get("items")[-1], 3)  # 最后一次迭代的值

    def testExecuteDict(self):
        """测试字典类型的执行"""
        forBlock = self.createForBlock()
        forBlock.compile()
        varDict = self.createVarDict({"a": 1, "b": 2})
        result = forBlock.execute(context=self.context, args=varDict.getAllDict())
        
        self.assertTrue(forBlock.loopBlock.isExecuteCalled)
        self.assertIsInstance(result, Response)
        
        # 将 dict_keys 转换为 list 后比较
        actual_keys = list(forBlock.loopBlock.executeArgs.get("items").keys())
        expected_keys = ["a", "b"]
        self.assertEqual(sorted(actual_keys), sorted(expected_keys))

    def testExecuteEmptyIterable(self):
        """测试空迭代对象的执行"""
        testCases = [[], {}]
        forBlock = self.createForBlock()
        forBlock.compile()
        
        for items in testCases:
            with self.subTest(items=items):
                varDict = self.createVarDict(items)
                result = forBlock.execute(context=self.context, args=varDict.getAllDict())
                self.assertIsInstance(result, Response)

if __name__ == '__main__':
    unittest.main()