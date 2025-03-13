import os
import shutil
import logging
from typing import List, Callable
from milkie.functions.toolkits.toolkit import Toolkit
from milkie.functions.openai_function import OpenAIFunction
from milkie.global_context import GlobalContext

logger = logging.getLogger(__name__)

class FilesysToolkit(Toolkit):
    MaxFilesPerDirectory = 3  # 每个目录最多显示的文件数
    MaxTreeDepth = 3  # 树的最大深度

    def __init__(self, globalContext :GlobalContext=None):
        super().__init__(globalContext)

    def getName(self) -> str:
        return "FilesysToolkit"

    def createDirectory(
            self, 
            parentPath: str, 
            directoryName: str) -> str:
        """
        在指定的父路径(绝对路径)中创建一个新目录。

        Args:
            parentPath (str): 将要创建新目录的父目录路径。父目录必须为绝对路径。
            directoryName (str): 要创建的新目录的名称。

        Returns:
            str: 创建的新目录的完整路径或错误消息。
        """
        try:
            if not os.path.isabs(parentPath):
                return "Parent path must be an absolute path."

            fullPath = os.path.join(parentPath, directoryName)
            os.makedirs(fullPath, exist_ok=True)
            logger.info(f"Directory created successfully: {fullPath}")
            return f"Directory created successfully: {fullPath}"
        except Exception as e:
            errorMsg = f"Error creating directory: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def copyFiles(
            self, 
            sourcePath: str, 
            destinationPath: str, 
            nameFilter: str) -> str:
        """
        根据名称过滤器将文件从源路径复制到目标路径。

        Args:
            sourcePath (str): 源目录路径。不能以'.'开头
            destinationPath (str): 目标目录路径。如果是相对路径，则相对于sourcePath。
            nameFilter (str): 用于匹配文件名的过滤器（不区分大小写）。如果为空字符串，则匹配所有。

        Returns:
            str: 指示复制文件数量的消息或错误消息。
        """
        return self._processFiles(sourcePath, destinationPath, nameFilter, shutil.copy2)

    def moveFiles(
            self, 
            sourcePath: str, 
            destinationPath: str, 
            nameFilter: str) -> str:
        """
        根据名称过滤器将文件从源路径移动到目标路径。

        Args:
            sourcePath (str): 源目录路径。不能以'.'开头
            destinationPath (str): 目标目录路径。如果是相对路径，则相对于sourcePath。
            nameFilter (str): 用于匹配文件名的过滤器（不区分大小写）。如果为空字符串，则匹配所有。

        Returns:
            str: 指示移动文件数量的消息或错误消息。
        """
        return self._processFiles(sourcePath, destinationPath, nameFilter, shutil.move)

    def generateDirectoryTree(self, path: str) -> str:
        """
        生成指定路径下所有子文件和子目录的树表示。
        不显示以"."开头的文件，每个目录最多显示MaxFilesPerDirectory个文件。
        树的深度限制为MaxTreeDepth。

        Args:
            path (str): 要生成树表示的目录路径。

        Returns:
            str: 目录结构的树表示。
        """
        try:
            def generateTree(dirPath, prefix="", depth=0):
                if depth >= self.MaxTreeDepth:
                    return f"{prefix}...\n"

                tree = ""
                entries = os.listdir(dirPath)
                entries = [e for e in entries if not e.startswith('.')]  # 过滤掉以"."开头的文件和目录
                entries.sort()
                dirs = [e for e in entries if os.path.isdir(os.path.join(dirPath, e))]
                files = [e for e in entries if os.path.isfile(os.path.join(dirPath, e))]

                for i, entry in enumerate(dirs + files[:self.MaxFilesPerDirectory]):
                    fullPath = os.path.join(dirPath, entry)
                    isLast = i == len(dirs) + min(len(files), self.MaxFilesPerDirectory) - 1
                    tree += f"{prefix}{'└── ' if isLast else '├── '}{entry}\n"
                    if os.path.isdir(fullPath):
                        tree += generateTree(fullPath, prefix + ('    ' if isLast else '│   '), depth + 1)

                if len(files) > self.MaxFilesPerDirectory:
                    tree += f"{prefix}└── ... ({len(files) - self.MaxFilesPerDirectory} more files)\n"

                return tree

            if not os.path.exists(path):
                return f"Error: Path '{path}' does not exist."

            tree = f"{os.path.basename(path)}\n" + generateTree(path)
            logger.info(f"Directory tree generated successfully:\n{tree}")
            return tree
        except Exception as e:
            errorMsg = f"Error generating directory tree: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def removeEmptyDirectories(self, path: str) -> str:
        """
        发现并删除指定绝对路径下的所有空目录。

        Args:
            path (str): 要搜索空目录的绝对路径。

        Returns:
            str: 操作描述，包括已删除的空目录数量。
        """
        try:
            if not os.path.isabs(path):
                return f"Error: The provided path '{path}' is not an absolute path."

            removedCount = 0
            for root, dirs, files in os.walk(path, topdown=False):
                for dir in dirs:
                    dirPath = os.path.join(root, dir)
                    if not os.listdir(dirPath):  # 检查目录是否为空
                        os.rmdir(dirPath)
                        removedCount += 1
                        logger.info(f"Removed empty directory: {dirPath}")

            resultMsg = f"Successfully removed {removedCount} empty directories."
            logger.info(resultMsg)
            return resultMsg
        except Exception as e:
            errorMsg = f"Error removing empty directories: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def listFiles(self, directoryPath: str, nameFilter: str = "") -> List[str]:
        """
        列出指定目录及其子目录中所有文件的绝对路径，忽略以"."开头的文件，并可选择性地根据nameFilter进行过滤。

        Args:
            directoryPath (str): 开始列举的目录路径。
            nameFilter (str): 用于匹配文件名的过滤器（不区分大小写）。如果为空字符串，则匹配所有非隐藏文件。

        Returns:
            List[str]: 文件绝对路径的列表。
        """
        try:
            absolutePaths = []
            for root, dirs, files in os.walk(directoryPath):
                # 忽略以"."开头的目录
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if not file.startswith('.') and self._matchesNameFilter(file, nameFilter):
                        absolutePaths.append(os.path.abspath(os.path.join(root, file)))
            
            logger.info(f"Successfully listed {len(absolutePaths)} file paths in {directoryPath}")
            return absolutePaths
        except Exception as e:
            errorMsg = f"Error listing absolute paths: {str(e)}"
            logger.error(errorMsg)
            return []

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.createDirectory),
            OpenAIFunction(self.copyFiles),
            OpenAIFunction(self.generateDirectoryTree),
            OpenAIFunction(self.removeEmptyDirectories),
            OpenAIFunction(self.moveFiles),
            OpenAIFunction(self.listFiles),
        ]

    def _filterWalk(self, walked):
        """
        过滤掉只包含以'.'开头的项目的目录

        Args:
            walked: os.walk()的迭代器

        Yields:
            过滤后的(root, dirs, files)元组
        """
        for root, dirs, files in walked:
            # 从列表中移除隐藏目录和文件
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]
            
            # 只在有可见文件或目录时yield
            if dirs or files:
                yield root, dirs, files

    def _processFiles(
            self, 
            sourcePath: str, 
            destinationPath: str, 
            nameFilter: str,
            operation: Callable[[str, str], None]) -> str:
        """
        处理文件的通用方法。

        Args:
            sourcePath (str): 源目录路径。
            destinationPath (str): 目标目录路径。
            nameFilter (str): 用于匹配文件名的过滤器。
            operation (Callable[[str, str], None]): 要执行的操作(复制或移动)。

        Returns:
            str: 指示处理文件数量的消息或错误消息。
        """
        try:
            if sourcePath.startswith('.'):
                return "Source path cannot start with '.'."

            sourcePath = os.path.abspath(sourcePath)
            destinationPath = os.path.abspath(destinationPath)

            if sourcePath == destinationPath:
                return "Source path and destination path are the same. No operation needed."

            if not os.path.exists(destinationPath):
                os.makedirs(destinationPath)

            processedFiles = 0
            walked = self._filterWalk(os.walk(sourcePath))
            for root, _, files in walked:
                for file in files:
                    if self._matchesNameFilter(file, nameFilter):
                        sourceFile = os.path.join(root, file)
                        relPath = os.path.relpath(root, sourcePath)
                        
                        if relPath == '.':
                            destFile = os.path.join(destinationPath, file)
                        else:
                            destDir = os.path.normpath(os.path.join(destinationPath, relPath))
                            destFile = os.path.join(destDir, file)
                        
                        # 检查文件是否已经在目标路径中
                        if os.path.commonpath([sourceFile, destinationPath]) == destinationPath:
                            continue

                        # 检查源文件是否存在且目标文件不存在
                        if os.path.exists(sourceFile) and not os.path.exists(destFile):
                            os.makedirs(os.path.dirname(destFile), exist_ok=True)
                            
                            operation(sourceFile, destFile)
                            processedFiles += 1
                            logger.info(f"Processed file: {sourceFile} to {destFile}")
                        elif os.path.exists(destFile):
                            logger.info(f"Skipped file (already exists at destination): {sourceFile}")
                        else:
                            logger.info(f"Skipped file (source file does not exist): {sourceFile}")

            resultMsg = f"Successfully processed {processedFiles} files."
            logger.info(resultMsg)
            return resultMsg
        except Exception as e:
            errorMsg = f"Error processing files: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def _matchesNameFilter(self, name: str, nameFilter: str) -> bool:
        """
        确定文件或目录名是否匹配给定的名称过滤器。

        Args:
            name (str): 文件或目录名称。
            nameFilter (str): 名称过滤器。

        Returns:
            bool: 如果名称匹配过滤器，则为True；否则为False。
        """
        return not nameFilter or nameFilter.lower() in name.lower()

if __name__ == "__main__":
    filesysTools = FilesysToolkit()
    print(filesysTools.getDesc())