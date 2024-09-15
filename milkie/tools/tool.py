import os, io, sys, traceback, logging
from abc import ABC, abstractmethod

from milkie.context import Context
from milkie.global_context import GlobalContext
from milkie.llm.inference import chat
from milkie.model_factory import ModelFactory
from milkie.response import Response
from milkie.settings import Settings

logger = logging.getLogger(__name__)

class Tool(ABC):
 
    def __str__(self) -> str:
        return f"""
        名称: {self.name()}
        描述: {self.describe()}
        """   

    @abstractmethod
    def name(self) -> str:
        pass
 
    @abstractmethod
    def describe(self, classname, ) -> str:
        pass   

    @abstractmethod
    def execute(self) -> dict:
        pass

from PIL import ImageGrab

class ScreenShot(Tool):
    """
    Take screenshot automatically
    """
    def __init__(self) -> None:
        self.defaultPathScreenshot = os.getenv("DEFAULT_PATH_SCREENSHOT")
    
    def name(self) -> str:
        return "ScreenShot"
    
    def describe(self) -> str:
        return """
        Name: ScreenShot
        Desc: make a screen shot and save it as ".jpg" file in `savepath` 
        Args:
            savepath (String): the saved path of the screenshot

        Returns:
            dict: A JSON object representing the execution result
        """
    
    def execute(self, savepath :str) -> Response:
        screenshot = ImageGrab.grab()
        if not savepath:
            savepath = self.defaultPathScreenshot
        screenshot.save(savepath)
        return Response()

class CodeContext:
    def __init__(self, query :str) -> None:
        self.query = query
        self.code = None
        self.output = None 
        self.error = None

    def setResult(
            self, 
            code :str, 
            output :str, 
            error :str) -> None:
        self.code = code
        self.output = output
        self.error = error

    def isSucc(self) -> bool:
        return self.output and not self.error
    
    def hasError(self) -> bool:
        return self.error

    def getOutput(self) -> str:
        return self.output.strip()

class Coder(Tool):

    def name(self) -> str:
        return "python自动编程工具"

    def describe(self) -> str:
        return "能够根据要求自动生成 Python 代码解决问题"

    def execute(
            self, 
            context :Context, 
            query :str) -> Response:
        codeContext = CodeContext(query)
        self._genAndExecCode(context, codeContext)
        if not codeContext.isSucc():
            self._genAndExecCode(context, codeContext)
        
        if codeContext.isSucc():
            response = Response(response=codeContext.getOutput())
            if response.metadata:
                response.metadata = { "exception" : codeContext.error}
            return response
        return None

    def _genAndExecCode(
            self, 
            context :Context,
            codeContext :CodeContext):
        if not codeContext.hasError():
            prompt = f"请写一段 Python 代码完成下面的任务：{codeContext.query}"
        else :
            prompt = f"""
            已有代码：{codeContext.code}
            已有错误信息：{codeContext.error}
            请修改代码解决下面的问题：{codeContext.query}"""
            
        response = chat(
            llm=context.globalContext.settings.llmCode, 
            systemPrompt=None,
            prompt=prompt, 
            promptArgs={})

        #extract code part in response, assign to `code`
        import re
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response.response, re.DOTALL)
        if len(matches) == 1:
            code = matches[0]

        #execute `code` in virtual environment
        outputCapture = io.StringIO()

        output = None
        exceptionOutput = None
        try:
            sys.stdout = outputCapture
            exec(code.replace("__main__", __name__), globals())
            output = outputCapture.getvalue()
        except Exception as e:
            exceptionOutput = traceback.format_exc()
        except SystemExit as e:
            exceptionOutput = f"SystemExit: {e}"
        finally:
            sys.stdout = sys.__stdout__
            logger.info(f"output[{output}] exception[{exceptionOutput}]")

        codeContext.setResult(code, output, exceptionOutput)
        context.reqTrace.set("codeResp", response.response)

class LLM(Tool):

    def name(self) -> str:
        return "大语言模型"

    def describe(self) -> str:
        return "能够针对一段 context 进行深度语义理解，生成对应的回答"

    def execute(
            self, 
            context :Context, 
            query :str) -> Response:
        prompt = "请使用你强大的理解能力解决下面的问题：%s"
        response = chat(
            llm=context.globalContext.settings.llm, 
            systemPrompt=None,
            prompt=prompt % query, 
            promptArgs={})
        context.reqTrace.set("llmResp", response.response)
        return response

class ToolSet:
    def __init__(self, tools :list) -> None:
        self.tools = tools

    def describe(self) -> str:
        return "\n".join(str(tool) for tool in self.tools)  

    def choose(self, name :str) -> Tool:
        for tool in self.tools:
            if tool.name() in name:
                return tool
        return None

if __name__ == "__main__":
    from milkie.strategy import Strategy
    from playground.global_config import makeGlobalConfig

    globalConfig = makeGlobalConfig(Strategy.getStrategy("raw"))
    modelFactory = ModelFactory()
    globalContext = GlobalContext(globalConfig, modelFactory)
    coding = Coder()
    response = coding.execute(globalContext.settings, "计算不超过32的最大质数是多少")
    print(response)
