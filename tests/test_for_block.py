import unittest
from milkie.agent.for_block import ForBlock
from milkie.agent.llm_block import LLMBlock, Response
from milkie.context import Context

class MockLLMBlock(LLMBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compileCalled = False
        self.executeCalled = False

    def compile(self):
        self.compileCalled = True

    def execute(self, query=None, args={}, prevBlock=None):
        self.executeCalled = True
        return Response(respStr="Mock response", metadata={})

class TestForBlock(unittest.TestCase):

    def setUp(self):
        self.context = Context.createContext("config/global.yaml")

    def testParseForStatement(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}\nEND"
        )
        self.assertEqual(forBlock.loopVar, "item")
        self.assertEqual(forBlock.iterable, "items")
        self.assertEqual(forBlock.loopBody, "1. Process {item}\nEND")

    def testInvalidForStatement(self):
        with self.assertRaises(ValueError):
            ForBlock("invalid for statement", context=self.context)

    def testEmptyLoopBody(self):
        forBlock = ForBlock("FOR item in items:", context=self.context)
        self.assertEqual(forBlock.loopBody, "pass")

    def testCompile(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}\nEND",
            loopBlockClass=MockLLMBlock
        )
        forBlock.compile()
        self.assertTrue(forBlock.loopBlock.compileCalled)

    def testValidateList(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}"
        )
        result = forBlock.validate({"items": [1, 2, 3]})
        self.assertEqual(forBlock.loopType, list)
        self.assertEqual(result, [1, 2, 3])

    def testValidateDict(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}"
        )
        result = forBlock.validate({"items": {"a": 1, "b": 2}})
        self.assertEqual(forBlock.loopType, dict)
        self.assertEqual(result, {"a": 1, "b": 2})

    def testValidateInvalidType(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}"
        )
        with self.assertRaises(ValueError):
            forBlock.validate({"items": 42})

    def testExecuteList(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}",
            loopBlockClass=MockLLMBlock
        )
        forBlock.compile()
        result = forBlock.execute(args={"items": [1, 2, 3]})
        self.assertTrue(forBlock.loopBlock.executeCalled)
        self.assertIsInstance(result, Response)

    def testExecuteDict(self):
        forBlock = ForBlock(
            context=self.context,
            forStatement="FOR item in items:\n    1. Process {item}",
            loopBlockClass=MockLLMBlock
        )
        forBlock.compile()
        result = forBlock.execute(args={"items": {"a": 1, "b": 2}})
        self.assertTrue(forBlock.loopBlock.executeCalled)
        self.assertIsInstance(result, Response)

if __name__ == '__main__':
    unittest.main()