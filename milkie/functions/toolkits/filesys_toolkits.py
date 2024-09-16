import os
import shutil
import logging
from typing import List
from milkie.functions.toolkits.base import BaseToolkit
from milkie.functions.openai_function import OpenAIFunction

logger = logging.getLogger(__name__)

class FilesysToolKits(BaseToolkit):
    def __init__(self):
        super().__init__()

    def createDirectory(
            self, 
            parentPath: str, 
            directoryName: str) -> str:
        """
        在指定的父路径中创建一个新目录。

        Args:
            parentPath (str): 将要创建新目录的父目录路径。
            directoryName (str): 要创建的新目录的名称。

        Returns:
            str: 创建的新目录的完整路径或错误消息。
        """
        try:
            fullPath = os.path.join(parentPath, directoryName)
            os.makedirs(fullPath, exist_ok=True)
            logger.info(f"Directory created successfully: {fullPath}")
            return f"Directory created successfully: {fullPath}"
        except Exception as e:
            errorMsg = f"Error creating directory: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def moveFiles(
            self, 
            sourcePath: str, 
            destinationPath: str, 
            nameFilters: List[str], 
            fileExtensions: List[str]) -> str:
        """
        根据名称过滤器和文件扩展名将文件从源路径移动到目标路径。

        Args:
            sourcePath (str): 源目录路径。
            destinationPath (str): 目标目录路径。如果是相对路径，则相对于sourcePath。
            nameFilters (List[str]): 用于匹配文件名的过滤器列表（不区分大小写）。如果为空列表，则匹配所有。
            fileExtensions (List[str]): 要匹配的文件扩展名列表（例如 ['.txt', '.pdf']）。如果为空，则匹配所有文件类型。

        Returns:
            str: 指示移动文件数量的消息或错误消息。
        """
        try:
            sourcePath = os.path.abspath(sourcePath)
            if not os.path.isabs(destinationPath):
                destinationPath = os.path.abspath(os.path.join(sourcePath, destinationPath))
            else:
                destinationPath = os.path.abspath(destinationPath)

            if sourcePath == destinationPath:
                return "Source and destination paths are the same. No action needed."

            if not os.path.exists(destinationPath):
                os.makedirs(destinationPath)

            movedFiles = 0
            walked = os.walk(sourcePath)
            for root, _, files in walked:
                for file in files:
                    if self._shouldMoveItem(file, nameFilters, fileExtensions):
                        sourcefile = os.path.join(root, file)
                        relpath = os.path.relpath(root, sourcePath)
                        
                        # 确保我们不会在目标路径中创建重复的子目录
                        if relpath == '.':
                            destfile = os.path.join(destinationPath, file)
                        else:
                            destdir = os.path.normpath(os.path.join(destinationPath, relpath))
                            destfile = os.path.join(destdir, file)
                        
                        # 检查源文件是否存在，以及目标文件是否已经存在
                        if os.path.exists(sourcefile) and not os.path.exists(destfile):
                            os.makedirs(os.path.dirname(destfile), exist_ok=True)
                            
                            shutil.move(sourcefile, destfile)
                            movedFiles += 1
                            logger.info(f"Moved file: {sourcefile} to {destfile}")
                        elif os.path.exists(destfile):
                            logger.info(f"Skipped file (already exists at destination): {sourcefile}")
                        else:
                            logger.info(f"Skipped file (source file does not exist): {sourcefile}")

            resultMsg = f"Successfully moved {movedFiles} files."
            logger.info(resultMsg)
            return resultMsg
        except Exception as e:
            errorMsg = f"Error moving files: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def moveDirectories(
            self, 
            sourcePath: str, 
            destinationPath: str, 
            nameFilters: List[str]) -> str:
        """
        根据名称过滤器将目录从源路径移动到目标路径。

        Args:
            sourcePath (str): 源目录路径。
            destinationPath (str): 目标目录路径。如果是相对路径，则相对于sourcePath。
            nameFilters (List[str]): 用于匹配目录名的过滤器列表（不区分大小写）。如果为空列表，则匹配所有。

        Returns:
            str: 指示移动目录数量的消息或错误消息。
        """
        try:
            sourcePath = os.path.abspath(sourcePath)
            if not os.path.isabs(destinationPath):
                destinationPath = os.path.abspath(os.path.join(sourcePath, destinationPath))
            else:
                destinationPath = os.path.abspath(destinationPath)

            if sourcePath == destinationPath or destinationPath.startswith(sourcePath + os.path.sep):
                return "Source path is the same as or a parent of the destination path. No action needed."

            if not os.path.exists(destinationPath):
                os.makedirs(destinationPath)

            movedDirs = 0
            walked = os.walk(sourcePath, topdown=False)
            for root, dirs, _ in walked:
                for dir in dirs:
                    if self._shouldMoveItem(dir, nameFilters, []):
                        sourcedir = os.path.join(root, dir)
                        relpath = os.path.relpath(root, sourcePath)
                        
                        if relpath == '.':
                            destdir = os.path.join(destinationPath, dir)
                        else:
                            destparent = os.path.normpath(os.path.join(destinationPath, relpath))
                            destdir = os.path.join(destparent, dir)
                        
                        if os.path.exists(sourcedir) and not os.path.exists(destdir):
                            shutil.move(sourcedir, destdir)
                            movedDirs += 1
                            logger.info(f"Moved directory: {sourcedir} to {destdir}")
                        elif os.path.exists(destdir):
                            logger.info(f"Skipped directory (already exists at destination): {sourcedir}")
                        else:
                            logger.info(f"Skipped directory (source directory does not exist): {sourcedir}")

            resultMsg = f"Successfully moved {movedDirs} directories."
            logger.info(resultMsg)
            return resultMsg
        except Exception as e:
            errorMsg = f"Error moving directories: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def _shouldMoveItem(self, name: str, nameFilters: List[str], fileExtensions: List[str]) -> bool:
        """
        判断是否应该移动文件或目录。

        Args:
            name (str): 文件或目录名。
            nameFilters (List[str]): 名称过滤器列表。
            fileExtensions (List[str]): 文件扩展名列表。

        Returns:
            bool: 如果应该移动则返回 True，否则返回 False。
        """
        return (not nameFilters or any(filter.lower() in name.lower() for filter in nameFilters)) and \
               (not fileExtensions or any(name.lower().endswith(ext.lower()) for ext in fileExtensions))

    def generateDirectoryTree(self, path: str) -> str:
        """
        生成指定路径下所有子文件和子目录的树形表示。

        Args:
            path (str): 要生成树形表示的目录路径。

        Returns:
            str: 目录结构的树形表示。
        """
        try:
            def generateTree(dirPath, prefix=""):
                tree = ""
                entries = os.listdir(dirPath)
                entries.sort()
                for i, entry in enumerate(entries):
                    fullPath = os.path.join(dirPath, entry)
                    isLast = i == len(entries) - 1
                    tree += f"{prefix}{'└── ' if isLast else '├── '}{entry}\n"
                    if os.path.isdir(fullPath):
                        tree += generateTree(fullPath, prefix + ('    ' if isLast else '│   '))
                return tree

            if not os.path.exists(path):
                return f"Error: Path '{path}' does not exist."

            tree = f"{os.path.basename(path)}\n" + generateTree(path)
            logger.info(f"Successfully generated directory tree:\n{tree}")
            return tree
        except Exception as e:
            errorMsg = f"Error generating directory tree: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def removeEmptyDirectories(self, path: str) -> str:
        """
        发现并删除指定路径下的所有空目录。

        Args:
            path (str): 要搜索空目录的起始路径。

        Returns:
            str: 操作结果的描述，包括删除的空目录数量。
        """
        try:
            removed_count = 0
            for root, dirs, files in os.walk(path, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if not os.listdir(dir_path):  # 检查目录是否为空
                        os.rmdir(dir_path)
                        removed_count += 1
                        logger.info(f"Removed empty directory: {dir_path}")

            result_msg = f"Successfully removed {removed_count} empty directories."
            logger.info(result_msg)
            return result_msg
        except Exception as e:
            error_msg = f"Error removing empty directories: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.createDirectory),
            OpenAIFunction(self.moveFiles),
            OpenAIFunction(self.moveDirectories),
            OpenAIFunction(self.generateDirectoryTree),
            OpenAIFunction(self.removeEmptyDirectories),
        ]

if __name__ == "__main__":
    filesysTools = FilesysToolKits()
    print(filesysTools.getToolsDesc())