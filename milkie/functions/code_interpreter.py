import ast
from typing import Any, Dict, Optional
import logging
import traceback
from milkie.global_context import GlobalContext
from milkie.llm.step_llm import StepLLM
from milkie.log import DEBUG, ERROR, INFO, WARNING
from milkie.response import Response
from milkie.utils.data_utils import extractFromBlock, postRestoreVariablesInStr, wrapVariablesInStr
from milkie.vm.vm import VM

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
        self.maxAttempts = 1
        self.globalContext = globalContext

    def execute(
            self, 
            instruction: str, 
            varDict: Optional[Dict[str, Any]] = None, 
            vm: VM = None,
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
                codeRepr = code.replace('\n', '//')
                INFO(logger, f"execute code [{codeRepr}] model[{stepLLMCode.llm.model_name}]")
                result = vm.execPython(code, varDict=varDict)
                return VM.deserializePythonResult(result)
            except Exception as e:
                attempt += 1
                stepLLMCode.fail(args=varDict, **kwargs)
                errorContext = f"error[{str(e)}]\nstacktrace[{traceback.format_exc()}]"
                WARNING(logger, f"failed to execute code[{errorContext}]")
                
                if attempt >= self.maxAttempts:
                    return None