import unittest
from milkie.agent.llm_block import InstFlag

class TestInstFlag(unittest.TestCase):

    def test_end_flag(self):
        inst = InstFlag("Some instruction #END")
        self.assertEqual(inst.flag, InstFlag.Flag.END)
        self.assertIsNone(inst.outputSyntax)
        self.assertIsNone(inst.storeVar)

    def test_code_flag(self):
        inst = InstFlag("Some instruction #CODE => [论文链接abc, 发布日期123] -> varName")
        self.assertEqual(inst.flag, InstFlag.Flag.CODE)
        self.assertEqual(inst.outputSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.storeVar, "varName")

    def test_if_flag(self):
        inst = InstFlag("IF condition => [论文链接abc, 发布日期123] -> varName")
        self.assertEqual(inst.flag, InstFlag.Flag.IF)
        self.assertEqual(inst.outputSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.storeVar, "varName")

    def test_goto_flag(self):
        inst = InstFlag("Some instruction GOTO label => [论文链接abc, 发布日期123] -> varName")
        self.assertEqual(inst.flag, InstFlag.Flag.GOTO)
        self.assertEqual(inst.label, "label")
        self.assertEqual(inst.outputSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.storeVar, "varName")

    def test_no_flag(self):
        inst = InstFlag("Some instruction => [论文链接abc, 发布日期123] -> varName")
        self.assertEqual(inst.flag, InstFlag.Flag.NONE)
        self.assertEqual(inst.outputSyntax, "[论文链接abc, 发布日期123]")
        self.assertEqual(inst.storeVar, "varName")

if __name__ == '__main__':
    unittest.main()