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

    def createSubfolder(
            self, 
            parentPath: str, 
            folderName: str) -> str:
        """
        Create a subfolder in the specified parent path.

        Args:
            parentPath (str): The parent directory path where the subfolder will be created.
            folderName (str): The name of the subfolder to be created.

        Returns:
            str: The full path of the created subfolder or an error message.
        """
        try:
            fullPath = os.path.join(parentPath, folderName)
            os.makedirs(fullPath, exist_ok=True)
            logger.info(f"Subfolder created successfully: {fullPath}")
            return f"Subfolder created successfully: {fullPath}"
        except Exception as e:
            errorMsg = f"Error creating subfolder: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def moveFiles(
            self, 
            sourcePath: str, 
            destPath: str, 
            keywords: List[str], 
            fileTypes: List[str]) -> str:
        """
        Move files from source path to destination path based on keywords and file types.

        Args:
            sourcePath (str): The source directory path.
            destPath (str): The destination directory path.
            keywords (List[str]): List of keywords to match in file names (case-insensitive).
            fileTypes (List[str]): List of file extensions to match (e.g., ['.txt', '.pdf']).

        Returns:
            str: A message indicating the number of files moved or an error message.
        """
        try:
            if not os.path.exists(destPath):
                os.makedirs(destPath)

            movedFiles = 0
            for root, _, files in os.walk(sourcePath):
                for file in files:
                    if any(keyword.lower() in file.lower() for keyword in keywords) and \
                       any(file.lower().endswith(fileType.lower()) for fileType in fileTypes):
                        sourceFile = os.path.join(root, file)
                        destFile = os.path.join(destPath, file)
                        shutil.move(sourceFile, destFile)
                        movedFiles += 1
                        logger.info(f"Moved file: {sourceFile} to {destFile}")

            resultMsg = f"Successfully moved {movedFiles} files."
            logger.info(resultMsg)
            return resultMsg
        except Exception as e:
            errorMsg = f"Error moving files: {str(e)}"
            logger.error(errorMsg)
            return errorMsg

    def getFileTree(self, path: str) -> str:
        """
        Generate a tree representation of all subfiles and subfolders in the specified path.

        Args:
            path (str): The directory path to generate the tree representation for.

        Returns:
            str: Tree representation of the directory structure.
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

    def getTools(self) -> List[OpenAIFunction]:
        return [
            OpenAIFunction(self.createSubfolder),
            OpenAIFunction(self.moveFiles),
            OpenAIFunction(self.getFileTree),
        ]

if __name__ == "__main__":
    filesysTools = FilesysToolKits()
    print(filesysTools.getToolsDesc())