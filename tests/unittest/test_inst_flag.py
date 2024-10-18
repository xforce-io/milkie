import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from milkie.agent.llm_block.inst_flag import InstFlag, OutputSyntaxFormat
from milkie.agent.func_block import RepoFuncs

class TestInstFlag(unittest.TestCase):

    def setUp(self):
        self.repoFuncs = RepoFuncs()  # 创建一个空的 RepoFuncs 对象用于测试

    def testEndFlag(self):
        inst = InstFlag("Some instruction #RET", self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.RET)
        self.assertEqual(len(inst.getOutputSyntaxes()), 0)
        self.assertEqual(len(inst.getStoreVars()), 0)

    def testCodeFlag(self):
        inst = InstFlag("Some instruction #CODE => [论文链接abc, 发布日期123] -> varName", self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.CODE)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.getStoreVars()[0], "varName")

    def testIfFlag(self):
        inst = InstFlag("#IF condition => [论文链接abc, 发布日期123] -> varName", self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.IF)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.getStoreVars()[0], "varName")

    def testGotoFlag(self):
        inst = InstFlag("Some instruction #GOTO label => [论文链接abc, 发布日期123] -> varName", self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.GOTO)
        self.assertEqual(inst.label, "label")
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.getStoreVars()[0], "varName")

    def testNoFlag(self):
        inst = InstFlag("Some instruction => [论文链接abc, 发布日期123] -> varName", self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.NONE)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.getStoreVars()[0], "varName")

    def testPyFlag(self):
        inst = InstFlag(''' #PY ```from datetime import datetime; f"论文摘要-{source}-{{datetime.now().strftime('%Y-%m-%d')}}"``` -> paperTitleStr''', self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.PY)
        self.assertEqual(inst.getStoreVars()[0], "paperTitleStr")
        self.assertEqual(len(inst.getOutputSyntaxes()), 0)
        self.assertEqual(inst.getInstruction(), """from datetime import datetime; f"论文摘要-{source}-{{datetime.now().strftime('%Y-%m-%d')}}\"""")

    def testCallFlag(self):
        inst = InstFlag('#CALL @obj "some argument"', self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.CALL)
        self.assertEqual(inst.callObj, "obj")
        self.assertEqual(inst.callArg, "some argument")

    def testThoughtFlag(self):
        inst = InstFlag('#THOUGHT Some thought', self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.THOUGHT)
        self.assertEqual(inst.getInstruction(), "Some thought")

    def testDecomposeFlag(self):
        inst = InstFlag('#DECOMPOSE Some task', self.repoFuncs)
        self.assertEqual(inst.flag, InstFlag.Flag.DECOMPOSE)
        self.assertEqual(inst.getInstruction(), "Some task")

    def testMultipleFlagsError(self):
        with self.assertRaises(Exception):
            InstFlag('#RET #CODE Some instruction', self.repoFuncs)

    def testInvalidSyntax(self):
        with self.assertRaises(Exception):
            InstFlag('Some instruction -> var1 -> var2', self.repoFuncs)

    def testOnlyStoreVar(self):
        inst = InstFlag("Some instruction -> var1", self.repoFuncs)
        self.assertEqual(inst.getInstruction(), "Some instruction")
        self.assertEqual(len(inst.instOutput.outputStructs), 1)
        self.assertEqual(len(inst.getOutputSyntaxes()), 0)
        self.assertEqual(inst.getStoreVars(), ["var1"])

    def testOnlyOutputSyntax(self):
        inst = InstFlag("Some instruction => [output]", self.repoFuncs)
        self.assertEqual(inst.getInstruction(), "Some instruction")
        self.assertEqual(len(inst.instOutput.outputStructs), 1)
        self.assertEqual(len(inst.getOutputSyntaxes()), 1)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "[output]")
        self.assertEqual(len(inst.getStoreVars()), 0)

    def testMultipleOutputAndStore(self):
        inst = InstFlag("Some instruction => output1 -> var1 => output2 -> var2", self.repoFuncs)
        self.assertEqual(inst.getInstruction(), "Some instruction")
        self.assertEqual(len(inst.instOutput.outputStructs), 2)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "output1")
        self.assertEqual(inst.getOutputSyntaxes()[1].originalSyntax, "output2")
        self.assertEqual(inst.getStoreVars(), ["var1", "var2"])

    def testOutputSyntaxWithoutStoreVar(self):
        inst = InstFlag("Some instruction => output1 -> var1 => output2", self.repoFuncs)
        self.assertEqual(inst.getInstruction(), "Some instruction")
        self.assertEqual(len(inst.instOutput.outputStructs), 2)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "output1")
        self.assertEqual(inst.getOutputSyntaxes()[1].originalSyntax, "output2")
        self.assertEqual(inst.getStoreVars(), ["var1"])

    def testRegexOutputSyntax(self):
        inst = InstFlag("Some instruction => r'\\d+' / \"Error: No number found\" -> var1", self.repoFuncs)
        outputSyntax = inst.getOutputSyntaxes()[0]
        self.assertEqual(outputSyntax.format, OutputSyntaxFormat.REGEX)
        self.assertIsNotNone(outputSyntax.regExpr)
        self.assertEqual(outputSyntax.errorMessage, "Error: No number found")

    def testExtractOutputSyntax(self):
        inst = InstFlag("Some instruction => e'Name: (.+)' / \"Error: No name found\" -> var1", self.repoFuncs)
        outputSyntax = inst.getOutputSyntaxes()[0]
        self.assertEqual(outputSyntax.format, OutputSyntaxFormat.EXTRACT)
        self.assertEqual(outputSyntax.extractPattern, "Name: (.+)")
        self.assertEqual(outputSyntax.errorMessage, "Error: No name found")

    def testConsecutiveStoreVarsError(self):
        with self.assertRaises(Exception):
            InstFlag("Some instruction -> var1 -> var2", self.repoFuncs)

    def testValidMultipleOutputAndStore(self):
        inst = InstFlag("Some instruction => output1 -> var1 => output2 -> var2", self.repoFuncs)
        self.assertEqual(inst.getInstruction(), "Some instruction")
        self.assertEqual(len(inst.instOutput.outputStructs), 2)
        self.assertEqual(inst.getOutputSyntaxes()[0].originalSyntax, "output1")
        self.assertEqual(inst.getOutputSyntaxes()[1].originalSyntax, "output2")
        self.assertEqual(inst.getStoreVars(), ["var1", "var2"])

    def testComplexCallFlag(self):
        instruction = '''#CALL @complexFunction "complex argument" => r'(\d+)' / "Error: No number found" -> var1 => e'Name: (.+)' / "Error: No name found" -> var2 => r'Age: (\d+)' / "Error: No age found" -> var3 => e'Email: (.+@.+\..+)' / "Error: No email found" -> var4'''
        inst = InstFlag(instruction, self.repoFuncs)

        # 检查基本属性
        self.assertEqual(inst.flag, InstFlag.Flag.CALL)
        self.assertEqual(inst.callObj, "complexFunction")
        self.assertEqual(inst.callArg, "complex argument")

        # 检查 InstOutput
        self.assertEqual(len(inst.instOutput.outputStructs), 4)

        # 检查第一个 OutputStruct
        self.assertEqual(inst.instOutput.outputStructs[0].storeVar, "var1")
        self.assertEqual(inst.instOutput.outputStructs[0].outputSyntax.format, OutputSyntaxFormat.REGEX)
        self.assertEqual(inst.instOutput.outputStructs[0].outputSyntax.regExpr.pattern, r'(\d+)')
        self.assertEqual(inst.instOutput.outputStructs[0].outputSyntax.errorMessage, "Error: No number found")

        # 检查第二个 OutputStruct
        self.assertEqual(inst.instOutput.outputStructs[1].storeVar, "var2")
        self.assertEqual(inst.instOutput.outputStructs[1].outputSyntax.format, OutputSyntaxFormat.EXTRACT)
        self.assertEqual(inst.instOutput.outputStructs[1].outputSyntax.extractPattern, 'Name: (.+)')
        self.assertEqual(inst.instOutput.outputStructs[1].outputSyntax.errorMessage, "Error: No name found")

        # 检查第三个 OutputStruct
        self.assertEqual(inst.instOutput.outputStructs[2].storeVar, "var3")
        self.assertEqual(inst.instOutput.outputStructs[2].outputSyntax.format, OutputSyntaxFormat.REGEX)
        self.assertEqual(inst.instOutput.outputStructs[2].outputSyntax.regExpr.pattern, r'Age: (\d+)')
        self.assertEqual(inst.instOutput.outputStructs[2].outputSyntax.errorMessage, "Error: No age found")

        # 检查第四个 OutputStruct
        self.assertEqual(inst.instOutput.outputStructs[3].storeVar, "var4")
        self.assertEqual(inst.instOutput.outputStructs[3].outputSyntax.format, OutputSyntaxFormat.EXTRACT)
        self.assertEqual(inst.instOutput.outputStructs[3].outputSyntax.extractPattern, 'Email: (.+@.+\..+)')
        self.assertEqual(inst.instOutput.outputStructs[3].outputSyntax.errorMessage, "Error: No email found")

        # 检查 getOutputSyntaxes 和 getStoreVars 方法
        self.assertEqual(len(inst.getOutputSyntaxes()), 4)
        self.assertEqual(inst.getStoreVars(), ["var1", "var2", "var3", "var4"])

    def testComplexExtractOutputSyntax(self):
        inst = InstFlag("Some instruction => e'时间/地点/历史人物' / \"收到的时间不太对哦\" -> var1", self.repoFuncs)
        outputSyntax = inst.getOutputSyntaxes()[0]
        self.assertEqual(outputSyntax.format, OutputSyntaxFormat.EXTRACT)
        self.assertEqual(outputSyntax.extractPattern, "时间/地点/历史人物")
        self.assertEqual(outputSyntax.errorMessage, "收到的时间不太对哦")

    def testComplexRegexOutputSyntax(self):
        inst = InstFlag("Some instruction => r'\\d+/\\d+/\\d+' / \"日期格式不正确\" -> var1", self.repoFuncs)
        outputSyntax = inst.getOutputSyntaxes()[0]
        self.assertEqual(outputSyntax.format, OutputSyntaxFormat.REGEX)
        self.assertEqual(outputSyntax.regExpr.pattern, r'\d+/\d+/\d+')
        self.assertEqual(outputSyntax.errorMessage, "日期格式不正确")

if __name__ == '__main__':
    unittest.main()
