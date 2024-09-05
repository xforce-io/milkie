import unittest
from milkie.agent.llm_block import TaskEngine, AnalysisResult

class TestTaskEngine(unittest.TestCase):

    def test_decompose_task_single_instruction(self):
        task = "请帮我计算下，1 和 2 的平均数是多少"
        result = TaskEngine._decomposeTask(task)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.result, AnalysisResult.Result.DECOMPOSE)
        self.assertEqual(len(result.instructions), 1)
        self.assertEqual(result.instructions[0], ("1", "请帮我计算下，1 和 2 的平均数是多少"))

    def test_decompose_task_multiple_instructions(self):
        task = """
        请执行任务 
        Bob. 生成个随机数 
            这个随机数需要时奇数
            返回这个数
        Alice. #IF 如果上一步结果是奇数，跳到 三，
            如果是偶数，跳到 4 
        Cath. 讲个笑话 
            #END
        Dave. #CODE
            用结果写一篇短文
        """
        result = TaskEngine._decomposeTask(task)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.result, AnalysisResult.Result.DECOMPOSE)
        self.assertEqual(len(result.instructions), 4)
        
        expected_instructions = [
            ("Bob", "生成个随机数//这个随机数需要时奇数//返回这个数"),
            ("Alice", "#IF 如果上一步结果是奇数，跳到 三，//如果是偶数，跳到 4"),
            ("Cath", "讲个笑话//#END"),
            ("Dave", "#CODE//用结果写一篇短文"),
        ]
        
        for i, (label, instruction) in enumerate(result.instructions):
            self.assertEqual((label, instruction), expected_instructions[i])

    def test_decompose_task_numbered_instructions(self):
        task = """
        请执行任务 
        1. 生成个随机数 
        二、 #IF 如果上一步结果是奇数，跳到 三，
            如果是偶数，跳到 4 
        三. 讲个笑话 #END
        四、 #CODE
            用结果写一篇短文
        """
        result = TaskEngine._decomposeTask(task)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.result, AnalysisResult.Result.DECOMPOSE)
        self.assertEqual(len(result.instructions), 4)
        
        expected_instructions = [
            ("1", "生成个随机数"),
            ("二", "#IF 如果上一步结果是奇数，跳到 三，//如果是偶数，跳到 4"),
            ("三", "讲个笑话 #END"),
            ("四", "#CODE//用结果写一篇短文"),
        ]
        
        for i, (label, instruction) in enumerate(result.instructions):
            self.assertEqual((label, instruction), expected_instructions[i])

if __name__ == '__main__':
    unittest.main()