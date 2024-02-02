import os
from abc import ABC, abstractmethod

class Tool(ABC):
    
    @abstractmethod
    def execute(self) -> dict:
        pass

    @abstractmethod
    def describe(self, classname, ) -> str:
        pass

from PIL import ImageGrab

class ScreenShot(Tool):
    """
    Take screenshot automatically
    """
    def __init__(self) -> None:
        self.defaultPathScreenshot = os.getenv("DEFAULT_PATH_SCREENSHOT")
    
    def describe(self) -> str:
        return """
        Name: ScreenShot
        Desc: Execute the tool
        Args:
            savepath (String): the saved path of the screenshot

        Returns:
            dict: A JSON object representing the execution result
        """
    
    def execute(self, savepath :str) -> dict:
        screenshot = ImageGrab.grab()
        if not savepath:
            savepath = self.defaultPathScreenshot
        screenshot.save(savepath)
        return {
            "errno":0
        }

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

class OCR(Tool):
    """
    OCR tool
    """
    def __init__(self) -> None:
        self.defaultPathScreenshot = os.getenv("DEFAULT_PATH_SCREENSHOT")
    
    def describe(self) -> str:
        return """
        Name: OCR
        Desc: Execute the tool
        Args:
            savepath (String): the saved path of the screenshot

        Returns:
            dict: A JSON object representing the execution result
        """
   
    def execute(self, savepath :str) -> dict:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        if not savepath:
            savepath = self.defaultPathScreenshot
        result = ocr.ocr(savepath, cls=True)
        return {
            "errno" : 0,
            "result" : result
        }