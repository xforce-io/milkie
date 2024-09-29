import unittest
from milkie.agent.func_block import FuncBlock
from milkie.context import Context
from milkie.config.constant import KeywordForStart

class TestFuncBlock(unittest.TestCase):
    def setUp(self):
        self.context = Context.create("config/global.yaml")

    def testSimpleFunction(self):
        funcDef = """
        DEF simpleFunc(param1, param2):
            LLM_BLOCK Some simple task
        END
        """
        funcBlock = FuncBlock(funcDef, context=self.context)
        funcBlock.compile()

        self.assertEqual(funcBlock.funcName, "simpleFunc")
        self.assertEqual(funcBlock.params, ["param1", "param2"])
        self.assertIsNotNone(funcBlock.flowBlock)

    def testFunctionWithForLoop(self):
        funcDef = f"""
        DEF complexFunc(papers, emailAddr):
            LLM_BLOCK Prepare email
            {KeywordForStart} paperLink in papers:
                LLM_BLOCK Process each paper
            END -> resp
            LLM_BLOCK Finalize email
        END
        """
        funcBlock = FuncBlock(funcDef, context=self.context)
        funcBlock.compile()

        self.assertEqual(funcBlock.funcName, "complexFunc")
        self.assertEqual(funcBlock.params, ["papers", "emailAddr"])
        self.assertIsNotNone(funcBlock.flowBlock)

        # 检查 FOR 循环是否被正确处理
        self.assertIn(KeywordForStart, funcBlock.flowBlock.flowCode)
        self.assertIn("END -> resp", funcBlock.flowBlock.flowCode)

    def testNestedForLoops(self):
        funcDef = f"""
        DEF nestedFunc(data):
            LLM_BLOCK Prepare processing
            {KeywordForStart} category in data:
                LLM_BLOCK Process category
                {KeywordForStart} item in category:
                    LLM_BLOCK Process item
                END -> itemResp
            END -> categoryResp
            LLM_BLOCK Finalize processing
        END
        """
        funcBlock = FuncBlock(funcDef, context=self.context)
        funcBlock.compile()

        self.assertEqual(funcBlock.funcName, "nestedFunc")
        self.assertEqual(funcBlock.params, ["data"])
        self.assertIsNotNone(funcBlock.flowBlock)

        # 检查嵌套的 FOR 循环是否被正确处理
        flowCode = funcBlock.flowBlock.flowCode
        self.assertEqual(flowCode.count(KeywordForStart), 2)
        self.assertEqual(flowCode.count("END ->"), 2)

    def testInvalidFunctionDefinition(self):
        funcDef = """
        INVALID simpleFunc(param1, param2):
            LLM_BLOCK Some simple task
        END
        """
        funcBlock = FuncBlock(funcDef, context=self.context)
        with self.assertRaises(SyntaxError):
            funcBlock.compile()

if __name__ == '__main__':
    unittest.main()