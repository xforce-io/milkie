import os, io, sys, traceback
from abc import ABC, abstractmethod

from llama_index.core import Response

from milkie.global_context import GlobalContext
from milkie.llm.inference import chat
from milkie.model_factory import ModelFactory
from milkie.settings import Settings

class Tool(ABC):
    
    @abstractmethod
    def name() -> str:
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
    
    def name() -> str:
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

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

class OCR(Tool):
    """
    OCR tool
    """
    def __init__(self) -> None:
        self.defaultPathScreenshot = os.getenv("DEFAULT_PATH_SCREENSHOT")
    
    def name() -> str:
        return "OCR"
    
    def describe(self) -> str:
        return """
        Name: OCR
        Desc: Execute the tool
        Args:
            savepath (String): the saved path of the screenshot

        Returns:
            dict: A JSON object representing the execution result
        """
   
    def execute(self, savepath :str) -> Response:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        if not savepath:
            savepath = self.defaultPathScreenshot
        result = ocr.ocr(savepath, cls=True)
        return Response(response=result)

class Coding(Tool):

    def name() -> str:
        return "Coding"

    def describe(self) -> str:
        return """
        Name: Coding
        Desc: Execute the tool
        """

    def execute(
            self, 
            settings :Settings, 
            query :str) -> Response:
        response = chat(settings.llmCode, query, {})

        #extract code part in response, assign to `code`
        code = response

        #execute `code` in virtual environment
        outputCapture = io.StringIO()
        originalStdout = sys.stdout

        output = None
        exceptionOutput = None
        try:
            sys.stdout = outputCapture
            exec(code)
            output = outputCapture.getvalue()
        except Exception as e:
            exceptionOutput = traceback.format_exc()
        finally:
            sys.stdout = originalStdout

        response = Response()
        response.response = output
        if response.metadata:
            response.metadata = { "exception" : exceptionOutput }
        return response

if __name__ == "__main__":
    from milkie.strategy import Strategy
    from playground.global_config import makeGlobalConfig

    globalConfig = makeGlobalConfig(Strategy.getStrategy("raw"))
    modelFactory = ModelFactory()
    globalContext = GlobalContext(globalConfig, modelFactory)
    coding = Coding()
    response = coding.execute(globalContext.settings, "计算不超过32的最大质数是多少")
    print(response)