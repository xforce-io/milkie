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

import pyautogui

class MouseMove(Tool):
    """
    Move mouse
    """
    def __init__(self) -> None:
        pass

    def describe(self) -> str:
        return """
        Name: MouseMove
        Desc: Move the mouse
        Args:
            x, y (int): coordinate of the mouse
            num_seconds (int): the duration of the mouse moving

        Returns:
            dict: A JSON object representing the execution result
        """
        
    def execute(self, x :int, y :int, num_seconds :int = 0.5):
        pyautogui.moveTo(x, y, duration=num_seconds)
        return {
            "errno" : 0
        }

class Typein(Tool):
    """
    Type in
    """
    def __init__(self) -> None:
        pass

    def execute(self, text :str, interval :int = 0.25):
        """
        Name: Typein
        Desc: Type in the text
        Args:
            text (str): the text to be typed in
            interval (int): the interval between each character

        Returns:
            dict: A JSON object representing the execution result
        """

        pyautogui.typewrite(text, interval=interval)
        return {
            "errno" : 0
        }

class Press(Tool):
    """
    Press the key
    """
    def __init__(self) -> None:
        pass

    def describe(self) -> str:
        return """
        Name: Press
        Desc: Press the key
        Args:
            key (str): the key to be pressed

        Returns:
            dict: A JSON object representing the execution result
        """

    def execute(self, key :str="enter"):
        pyautogui.press(key)
        return {
            "errno" : 0
        }