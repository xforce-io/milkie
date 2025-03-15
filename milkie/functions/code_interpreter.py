from typing import Any, Dict, Optional
import logging
import traceback
from milkie.context import Context
from milkie.functions.import_white_list import WhiteListImport, addPreImport
from milkie.global_context import GlobalContext
from milkie.interpreter.internal_python_interpreter import InternalPythonInterpreter
from milkie.llm.step_llm import StepLLM
from milkie.log import DEBUG, ERROR, INFO, WARNING
from milkie.response import Response
from milkie.utils.data_utils import extractFromBlock, postRestoreVariablesInStr, wrapVariablesInStr
import json

logger = logging.getLogger(__name__)

class StepLLMCode(StepLLM):
    def __init__(
            self, 
            globalContext: GlobalContext, 
            instruction: str, 
            prevResult: str,
            noCache: bool,
            errorContext: Optional[str] = None):
        super().__init__(
            globalContext=globalContext, 
            llm=globalContext.settings.getLLMCode(noCache=noCache))
        self.instruction = wrapVariablesInStr(instruction)
        self.prevResult = prevResult
        self.errorContext = errorContext
        
    def makePrompt(
            self, 
            useTool: bool = False, 
            args: dict = {}, 
            **kwargs) -> str:
        prompt = self.prevResult + f"""
        请根据指令生成Python代码，要求如下：
        (1)如果要调用'print(X)'或'return X'，请使用'return_value = X'替代
        (2)生成实际执行的代码，而不只是定义函数
        
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

    def formatResult(self, result: Response, **kwargs) -> str:
        return result.respStr.strip()

class CodeInterpreter:

    def __init__(self, globalContext: GlobalContext):
        self.interpreter = InternalPythonInterpreter(import_white_list=WhiteListImport)
        self.maxAttempts = 1
        self.globalContext = globalContext

    def execute(
            self, 
            instruction: str, 
            varDict: Optional[Dict[str, Any]] = None, 
            **kwargs) -> Any:
        attempt = 0
        errorContext = ""
        while attempt < self.maxAttempts:
            stepLLMCode = StepLLMCode(
                globalContext=self.globalContext, 
                instruction=instruction, 
                prevResult="",
                errorContext=errorContext,
                noCache=kwargs.get("no_cache", False) if attempt == 0 else True)
 
            try:
                code = stepLLMCode.completionAndFormat(
                    args=varDict,
                    **kwargs)
                code = extractFromBlock("python", code)
                code = postRestoreVariablesInStr(code, varDict)
                code = addPreImport(code)
                
                codeRepr = code.replace('\n', '//')
                INFO(logger, f"execute code [{codeRepr}] model[{stepLLMCode.llm.model_name}]")
                result = self.interpreter.run(code, code_type="python3", varDict=varDict)
                return result

            except Exception as e:
                attempt += 1
                stepLLMCode.fail(args=varDict, **kwargs)
                errorContext = f"error[{str(e)}]\nstacktrace[{traceback.format_exc()}]"
                WARNING(logger, f"failed to execute code[{errorContext}]")
                
                if attempt >= self.maxAttempts:
                    return None
    
    def executeCode(self, code: str, varDict: Optional[Dict[str, Any]] = None) -> Any:
        """执行代码，处理特殊字符和转义"""
        try:
            # 添加日志以便调试
            DEBUG(logger, f"Executing code: {code}")
            
            # 执行代码
            return self.interpreter.run(code.replace('\\"', ''), code_type="python", varDict=varDict)
            
        except Exception as e:
            ERROR(logger, f"Failed to execute code: {code}")
            ERROR(logger, f"Error type: {type(e)}")
            ERROR(logger, f"Error message: {str(e)}")
            
            # 如果是 JSON 解析错误，打印相关位置的内容
            if isinstance(e, json.JSONDecodeError):
                error_pos = e.pos
                context_start = max(0, error_pos - 50)
                context_end = min(len(code), error_pos + 50)
                context = code[context_start:context_end]
                ERROR(logger, f"JSON error context: {context}")
                ERROR(logger, f"Error position: {error_pos}")
                
            raise RuntimeError(f"Code execution failed: {str(e)}")

if __name__ == "__main__":
    context = Context.create("config/global.yaml")
    #codeInterpreter = CodeInterpreter(context.globalContext)
    #print(codeInterpreter.interpreter.run("print(random.randint(1, 10))", code_type="python3"))
    import pdb; pdb.set_trace()
    print(CodeInterpreter(context.globalContext).execute("发送内容‘hello’到我的邮箱 pengxu@aishu.cn"))