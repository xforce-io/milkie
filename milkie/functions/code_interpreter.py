from typing import Any, Dict, Optional
import logging
import traceback
from milkie.context import Context
from milkie.functions.import_white_list import WhiteListImport, addPreImport
from milkie.global_context import GlobalContext
from milkie.interpreter.internal_python_interpreter import InternalPythonInterpreter
from milkie.llm.step_llm import StepLLM
from milkie.response import Response

logger = logging.getLogger(__name__)

class StepLLMCode(StepLLM):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            instruction: str, 
            prevResult: str,
            errorContext: Optional[str] = None):
        super().__init__(globalContext, None, codeModel=True)
        self.instruction = instruction
        self.prevResult = prevResult
        self.errorContext = errorContext

    def makePrompt(self, useTool: bool = False, args: dict = {}, **kwargs) -> str:
        prompt = self.prevResult + f"""
        请根据指令生成Python代码，要求如下：
        （1）请不要调用'return'
        （2）如果要调用'print(X)'，请使用'return_value = str(X)'替代
        
        指令: {self.instruction}
        """
        
        if self.errorContext:
            prompt += f"""
            上次执行出现错误,请修正:
            {self.errorContext}
            """
        
        prompt += """
        请直接根据指令翻译成可执行的Python代码,不需要其他解释。
        """
        return prompt

    def formatResult(self, result: Response) -> str:
        return result.respStr.strip()

class CodeInterpreter:

    def __init__(self, globalContext: GlobalContext):
        self.interpreter = InternalPythonInterpreter(import_white_list=WhiteListImport)
        self.maxAttempts = 2
        self.globalContext = globalContext

    def execute(self, instruction: str, varDict: Optional[Dict[str, Any]] = None) -> str:
        attempt = 0
        errorContext = ""
        while attempt < self.maxAttempts:
            try:
                stepLLMCode = StepLLMCode(
                    self.globalContext, 
                    instruction, 
                    errorContext)
                code = stepLLMCode.completionAndFormat()
                code = code.replace("```python", "").replace("```", "")
                code = addPreImport(code)
                
                codeRepr = code.replace('\n', '//')
                logger.info(f"execute code [{codeRepr}]")
                result = self.interpreter.run(code, code_type="python3", varDict=varDict)
                return str(result)

            except Exception as e:
                attempt += 1
                errorContext = f"error[{str(e)}]\nstacktrace[{traceback.format_exc()}]"
                logger.warning(f"failed to execute code[{errorContext}]")
                
                if attempt >= self.maxAttempts:
                    return f"执行失败。最后一次错误: {errorContext}"
    
    def executeCode(self, code: str, varDict: Optional[Dict[str, Any]] = None) -> str:
        return self.interpreter.run(code, code_type="python3", varDict=varDict)

if __name__ == "__main__":
    context = Context.create("config/global.yaml")
    #codeInterpreter = CodeInterpreter(context.globalContext)
    #print(codeInterpreter.interpreter.run("print(random.randint(1, 10))", code_type="python3"))
    import pdb; pdb.set_trace()
    print(CodeInterpreter(context.globalContext).execute("发送内容‘hello’到我的邮箱 pengxu@aishu.cn"))