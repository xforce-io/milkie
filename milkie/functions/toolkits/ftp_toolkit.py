from milkie.functions.toolkits.toolkit import Toolkit
from milkie.global_context import GlobalContext
from typing import List
from milkie.functions.openai_function import OpenAIFunction

class FTPToolkit(Toolkit):
    def __init__(self, globalContext: GlobalContext):
        super().__init__(globalContext)


    def getName(self) -> str:
        return "FTPToolkit"
    
    def uploadFile(self, localPath: str, remotePath: str) -> str:
        """
        上传本地文件到虚拟机中

        Args:
            localPath (str): 本地文件路径
            remotePath (str): 远程文件路径

        Returns:
            str: 上传是否成功
        """
        ret = self.globalContext.vm.uploadFile(localPath, remotePath)
        if ret:
            return "ok"
        else:
            return "failed"
    
    def downloadFile(self, remotePath: str, localPath: str) -> str:
        """
        从虚拟机中下载文件到本地

        Args:
            remotePath (str): 远程文件路径
            localPath (str): 本地文件路径

        Returns:
            str: 下载是否成功
        """
        ret = self.globalContext.vm.downloadFile(remotePath, localPath)
        if ret:
            return "ok"
        else:
            return "failed"
    
    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.uploadFile),
            OpenAIFunction(self.downloadFile)
        ]
