import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from milkie.context import Context
import unittest
from tests.integration_test.base_test import BaseIntegrationTest
from milkie.agent.flow_block import FlowBlock
from milkie.functions.toolkits.filesys_toolkit import FilesysToolkit

class FilesysIntegrationTest(BaseIntegrationTest):
    def setUp(self):
        super().setUp()

        self.context = Context.create("config/global_test.yaml")
        self.scenarioDir = self.getScenarioDataDir('filesys')
        self.prepairScenarioDataDir(self.scenarioDir)

        self.kgDir = os.path.join(self.scenarioDir, 'kg')
        self.createTestFiles()
    
    def tearDown(self):
        super().tearDown()
    
    def createTestFiles(self):
        files = [
            '知识图谱_介绍.txt',
            '知识图谱_应用.pdf',
            '机器学习.docx',
            '深度学习知识图谱.pptx',
            '自然语言处理.txt',
            '中国体育发展史.md',
            '中国足球为什么这么差.pptx',
            '隋唐历史.md',
            '明朝那些事儿.md',
        ]
        for file in files:
            with open(os.path.join(self.scenarioDir, file), 'w') as f:
                f.write(f"This is a test file: {file}")
    
    def testCopyKgFiles(self):
        self.prepairScenarioDataDir(self.kgDir)

        flowCode = """
        0. 获取目录{sourceDir}下的所有文件 -> allFiles
        1. #CODE 从下面文件中筛选出文件名包含"知识图谱"的文件 --{allFiles}-- -> kgFiles
        2. 将{kgFiles}中的文件复制到目录{destDir}
        """
        flowBlock = self._buildFlowBlock(flowCode)
        
        args = {
            "sourceDir": self.scenarioDir,
            "destDir": self.kgDir
        }
        flowBlock.execute(context=self.context, args=args)
        self._checkCopyKgFiles()

    def testCopyKgFilesThought(self):
        self.prepairScenarioDataDir(self.kgDir)

        flowCode = """
        1. #THOUGHT 筛选{sourceDir}下所有文件名包含"知识图谱"的文件，并复制到{destDir}下
        2. #DECOMPOSE
        """
        flowBlock = self._buildFlowBlock(flowCode, usePrevResult=True)
        
        args = {
            "sourceDir": self.scenarioDir,
            "destDir": self.kgDir
        }
        flowBlock.execute(context=self.context, args=args)
        self._checkCopyKgFiles()

    def testClassifyFiles(self):
        self.prepairScenarioDataDir(self.kgDir)
        
        flowCode = """
        0. 获取目录{sourceDir}下的所有文件 -> allFiles
        1. #CODE 从下面文件中筛选出文件名包含"知识图谱"的文件 --{allFiles}-- -> kgFiles
        2. 将{kgFiles}中的文件复制到目录{destDir}
        """

    def _buildFlowBlock(self, flowCode, usePrevResult=False):
        flowBlock = FlowBlock(
            flowCode=flowCode,
            context=self.context,
            toolkit=FilesysToolkit(self.context.globalContext),
            usePrevResult=usePrevResult
        )
        flowBlock.compile()
        return flowBlock

    def _checkCopyKgFiles(self):
        expectedFiles = [
            '知识图谱_介绍.txt',
            '知识图谱_应用.pdf',
            '深度学习知识图谱.pptx'
        ]
        copiedFiles = os.listdir(self.kgDir)
        
        self.assertEqual(set(expectedFiles), set(copiedFiles), "复制的文件不符合预期")
        
        for file in copiedFiles:
            sourcePath = os.path.join(self.scenarioDir, file)
            destPath = os.path.join(self.kgDir, file)
            self.assertTrue(os.path.exists(destPath), f"文件 {file} 未被复制")
            self.assertEqual(
                os.path.getsize(sourcePath),
                os.path.getsize(destPath),
                f"文件 {file} 的大小不一致"
            )

if __name__ == '__main__':
    unittest.main()