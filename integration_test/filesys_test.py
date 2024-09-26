import os
import shutil
import unittest
from base_test import BaseIntegrationTest
from milkie.agent.flow_block import FlowBlock
from milkie.functions.toolkits.filesys_toolkits import FilesysToolKits

class FilesysIntegrationTest(BaseIntegrationTest):
    def setUp(self):
        super().setUp()
        self.scenarioDir = self.getScenarioDataDir('filesys')
        self.kgDir = os.path.join(self.scenarioDir, 'kg')
        os.makedirs(self.kgDir, exist_ok=True)
        
        self.createTestFiles()
    
    def tearDown(self):
        super().tearDown()
    
    def createTestFiles(self):
        files = [
            '知识图谱_介绍.txt',
            '知识图谱_应用.pdf',
            '机器学习.docx',
            '深度学习知识图谱.pptx',
            '自然语言处理.txt'
        ]
        for file in files:
            with open(os.path.join(self.scenarioDir, file), 'w') as f:
                f.write(f"This is a test file: {file}")
    
    def testCopyKgFiles(self):
        # 清空目标目录
        shutil.rmtree(self.kgDir)
        os.makedirs(self.kgDir)
        
        # 定义 FlowBlock
        flowCode = """
        0. 获取目录{sourceDir}下的所有文件 -> allFiles
        1. #CODE 筛选出文件名包含"知识图谱"的文件 -> kgFiles
        2. 将{kgFiles}中的文件复制到目录{destDir}
        3. 获取目录{destDir}下的所有文件 -> copiedFiles
        """
        
        flowBlock = FlowBlock(
            flowCode=flowCode,
            toolkit=FilesysToolKits()
        )
        flowBlock.compile()
        
        # 执行 FlowBlock
        args = {
            "sourceDir": self.scenarioDir,
            "destDir": self.kgDir
        }
        result = flowBlock.execute(args=args)
        
        # 检查结果
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