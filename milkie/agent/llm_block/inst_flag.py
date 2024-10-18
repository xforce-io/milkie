from enum import Enum
import json
import logging
import re
from typing import Dict, List, Union
from milkie.agent.func_block import RepoFuncs
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.config.constant import *
from milkie.context import VarDict

logger = logging.getLogger(__name__)

class OutputSyntaxFormat(Enum):
    NORMAL = 1
    REGEX = 2
    EXTRACT = 3

class ResultOutputProcessSingle:
    def __init__(self, storeVar: str, output: str=None, errmsg: str=None):
        self.storeVar = storeVar
        self.output = output
        self.errmsg = errmsg
    
    def hasError(self) -> bool:
        return self.errmsg is not None

    def __str__(self) -> str:
        return f"ResultOutputProcessSingle(storeVar={self.storeVar}, output={self.output}, errmsg={self.errmsg})"

class ResultOutputProcess:
    def __init__(self):
        self._results: Dict[str, ResultOutputProcessSingle] = {}
        
    def addSuccess(self, storeVar: str, output: str):
        self._results[storeVar] = ResultOutputProcessSingle(storeVar, output, None)

    def addError(self, storeVar: str, errmsg: str):
        self._results[storeVar] = ResultOutputProcessSingle(storeVar, None, errmsg)

    def isSuccess(self, storeVar: str):
        return storeVar in self._results and not self._results[storeVar].hasError()

    def hasError(self) -> bool:
        return any(result.hasError() for result in self._results.values())

    def getErrMsgs(self) -> List[str]:
        return [self._results[storeVar].errmsg for storeVar in self._results if self._results[storeVar].hasError()]

    def setVarDict(self, varDict: VarDict):
        for storeVar, result in self._results.items():
            if result.hasError():
                continue

            if result.output.startswith("```json"):
                jsonStr = result.output.replace("```json", '').replace("```", '').replace("\n", '')
                try:
                    varDict.setGlobal(storeVar, json.loads(jsonStr))
                except Exception as e:
                    logger.error(f"Error parsing JSON: {e}")
                    varDict.setGlobal(storeVar, jsonStr)
            else:
                varDict.setGlobal(storeVar, result.output)
    
    def __str__(self) -> str:
        return f"ResultOutputProcess(results={self._results})"

class OutputSyntax:
    def __init__(self, syntax: str):
        self.originalSyntax = syntax.strip()
        self.format = self._determineFormat()
        self.regExpr = None
        self.extractPattern = None
        self.errorMessage = None
        self._parse()

    def _determineFormat(self) -> OutputSyntaxFormat:
        if self.originalSyntax.startswith("r'") or self.originalSyntax.startswith('r"'):
            return OutputSyntaxFormat.REGEX
        elif self.originalSyntax.startswith("e'") or self.originalSyntax.startswith('e"'):
            return OutputSyntaxFormat.EXTRACT
        else:
            return OutputSyntaxFormat.NORMAL

    def _parse(self):
        if self.format == OutputSyntaxFormat.NORMAL:
            return

        # 移除开头的 'r' 或 'e'
        content = self.originalSyntax[1:]

        # 查找最后一个未转义的引号
        quote_char = content[0]
        last_quote_index = -1
        i = 1
        while i < len(content):
            if content[i] == quote_char and content[i-1] != '\\':
                last_quote_index = i
            i += 1

        if last_quote_index == -1:
            raise ValueError("Invalid syntax: missing closing quote")

        pattern = content[1:last_quote_index]
        remaining = content[last_quote_index+1:].strip()

        if remaining.startswith('/'):
            self.errorMessage = remaining[1:].strip().strip('"\'')
        else:
            self.errorMessage = None

        if self.format == OutputSyntaxFormat.REGEX:
            self.regExpr = re.compile(pattern)
        else:  # EXTRACT
            self.extractPattern = pattern

    def getOutputSyntax(self):
        return re.sub(r'\{{2,}', '{', re.sub(r'\}{2,}', '}', self.originalSyntax))

    def processOutput(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: str) -> str:
        if self.format == OutputSyntaxFormat.REGEX:
            return self.regExpr.search(output).group(1)
        elif self.format == OutputSyntaxFormat.EXTRACT:
            return stepLLMExtractor.run(
                args={
                    "toExtract": self.extractPattern,
                    "text": output
                })
        else:
            return None

    def __str__(self) -> str:
        return f"OutputSyntax(originalSyntax={self.originalSyntax}, format={self.format}, regExpr={self.regExpr}, extractPattern={self.extractPattern}, errorMessage={self.errorMessage})"

class OutputStruct:
    def __init__(self, outputSyntax: OutputSyntax = None, storeVar: str = None):
        self.outputSyntax = outputSyntax
        self.storeVar = storeVar

    def processOutput(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: str) -> str:
        return self.outputSyntax.processOutput(stepLLMExtractor, output)

    def __str__(self) -> str:
        return f"OutputStruct(outputSyntax={self.outputSyntax}, storeVar={self.storeVar})"

class InstrOutput:
    def __init__(self):
        self.outputStructs: List[OutputStruct] = []
        self._processed = False
        self._currentResult: ResultOutputProcess = None

    def addOutputStruct(self, outputStruct: OutputStruct):
        self.outputStructs.append(outputStruct)
    
    def getNormalFormat(self) -> str:
        for outputStruct in self.outputStructs:
            if outputStruct.outputSyntax and outputStruct.outputSyntax.format == OutputSyntaxFormat.NORMAL:
                return outputStruct.outputSyntax.getOutputSyntax()
        return None
    
    def getCurrentResult(self):
        return self._currentResult

    def storeResultToVarDict(self, varDict: VarDict):
        if self._currentResult:
            self._currentResult.setVarDict(varDict)
    
    def processOutputAndStore(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: str,
            varDict: VarDict,
            retry: bool):
        if self._processed:
            return

        self._processOutput(stepLLMExtractor, output)
        if not self._currentResult.hasError():
            self._processed = True
            self.storeResultToVarDict(varDict)
        else:
            if not retry:
                raise RuntimeError(f"fail process output[{output}]")
    
    def isProcessed(self):
        return self._processed

    def hasError(self):
        return self._currentResult.hasError()

    def getErrMsgs(self):
        return self._currentResult.getErrMsgs()

    def _processOutput(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: str):
        if self._currentResult is None:
            self._currentResult = ResultOutputProcess()

        for outputStruct in self.outputStructs:
            if self._currentResult.isSuccess(outputStruct.storeVar):
                continue

            if outputStruct.outputSyntax:
                processedOutput = outputStruct.processOutput(stepLLMExtractor, output)
                if processedOutput is None or processedOutput.strip() == ExprNoInfoToExtract:
                    self._currentResult.addError(
                        outputStruct.storeVar,
                        outputStruct.outputSyntax.errorMessage)
                else:
                    self._currentResult.addSuccess(
                        outputStruct.storeVar,
                        processedOutput)
            else:
                self._currentResult.addSuccess(
                    outputStruct.storeVar,
                    output=output)

class InstFlag:
    class Flag(Enum):
        NONE = 1
        RET = 2
        CODE = 3
        IF = 4
        GOTO = 5
        PY = 6
        CALL = 7
        THOUGHT = 8
        DECOMPOSE = 9

    class TypeCall(Enum):
        STDIN = 1
        Agent = 2

    def __init__(
            self, 
            instruction: str,
            repoFuncs: RepoFuncs) -> None:
        self.flag = InstFlag.Flag.NONE
        self.label = None
        self.instOutput = InstrOutput()
        self.returnVal = False 
        self.instruction = instruction.strip()
        self.funcsToCall = []
        self.repoFuncs = repoFuncs
        self.typeCall = None
        self.callObj = None
        self.callArg = None

        self.parseInstruction()

    def getInstruction(self):
        return self.instruction

    def getInstrOutput(self):
        return self.instOutput

    def getNormalFormat(self) -> str:
        return self.instOutput.getNormalFormat()

    def getStoreVars(self):
        return [struct.storeVar for struct in self.instOutput.outputStructs if struct.storeVar]

    def parseInstruction(self):
        self._handleOutputAndStore()
        self._handleFlags()
        self._handleFunctions()

    def _handleOutputAndStore(self):
        parts = re.split(r'(=>|->)', self.instruction)
        self.instruction = parts[0].strip()
        
        outputSyntax = None
        lastSeparator = None
        for i in range(1, len(parts), 2):
            separator = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            if separator == "=>":
                outputSyntax = OutputSyntax(content) if content else None
                lastSeparator = separator
            elif separator == "->":
                if lastSeparator == "->" and outputSyntax is None:
                    raise Exception("Invalid syntax: consecutive store variables without output syntax")
                storeVar = content
                self.instOutput.addOutputStruct(OutputStruct(outputSyntax, storeVar))
                outputSyntax = None
                lastSeparator = separator
        
        # Handle the case where there's an outputSyntax without a following storeVar
        if outputSyntax:
            self.instOutput.addOutputStruct(OutputStruct(outputSyntax, None))

        # Check for invalid syntax
        if '->' in self.instruction:
            raise Exception("Invalid syntax: store variable in instruction")

    def _handleFunctions(self):
        if self.repoFuncs is not None:
            for funcName, funcBlock in self.repoFuncs.getAll().items():
                pattern = funcBlock.getFuncPattern()
                if pattern not in self.instruction:
                    continue
                
                paramsPattern = r'%s\(\s*([a-zA-Z0-9_,{}\s]+)\s*\)' % pattern
                paramsMatch = re.search(paramsPattern, self.instruction)
                if paramsMatch:
                    params = paramsMatch.group(1).strip()
                    params = params.replace(" ", "")
                    params = params.split(",")
                    funcBlock.setParams(params)
                    self.funcsToCall.append(funcBlock)
                    self.instruction = self.instruction.replace(paramsMatch.group(0), pattern)
                else:
                    raise SyntaxError(f"function[{funcName}] params not found")

    def _handleFlags(self):
        flagHandlers = {
            InstFlagRet: self._handleRetFlag,
            InstFlagCode: self._handleCodeFlag,
            InstFlagIf: self._handleIfFlag,
            InstFlagGoto: self._handleGotoFlag,
            InstFlagPy: self._handlePyFlag,
            InstFlagCall: self._handleCallFlag,
            InstFlagThought: self._handleThoughtFlag,
            InstFlagDecompose: self._handleDecomposeFlag
        }

        flagsFound = [flag for flag in flagHandlers.keys() if flag in self.instruction]

        if len(flagsFound) > 1:
            raise Exception("Multiple flags found in instruction")

        if flagsFound:
            flagHandlers[flagsFound[0]]()
        else:
            self.flag = InstFlag.Flag.NONE

    def _handleRetFlag(self):
        self.flag = InstFlag.Flag.RET
        self.returnVal = self.instruction.startswith(InstFlagRet)
        self.instruction = self.instruction.replace(InstFlagRet, "").strip()

    def _handleCodeFlag(self):
        self.flag = InstFlag.Flag.CODE
        self.instruction = self.instruction.replace(InstFlagCode, "")

    def _handleIfFlag(self):
        self.flag = InstFlag.Flag.IF

    def _handleGotoFlag(self):
        self.flag = InstFlag.Flag.GOTO
        gotoParts = self.instruction.split(InstFlagGoto)
        self.label = gotoParts[1].split()[0].strip()
        self.instruction = gotoParts[0].strip() + " ".join(gotoParts[1].split()[1:]).strip()

    def _handlePyFlag(self):
        self.flag = InstFlag.Flag.PY
        pyCode = self._extractPyCode()
        self.instruction = pyCode if pyCode else self.instruction.replace(InstFlagPy, "").strip()

    def _handleCallFlag(self):
        self.flag = InstFlag.Flag.CALL
        callParts = self.instruction.split(InstFlagCall)

        self.callObj = callParts[1].split()[0].strip()
        if self.callObj.startswith("@"):
            self.typeCall = InstFlag.TypeCall.Agent
            self.callObj = self.callObj[1:]
        else:
            raise Exception(f"Invalid call object: {self.callObj}")

        callArgPattern = r'%s\s+(["\'])((?:(?!\1).)*)\1' % re.escape(self.callObj)
        callArgMatch = re.search(callArgPattern, self.instruction)
        if callArgMatch:
            self.callArg = callArgMatch.group(2)
            self.instruction = self.instruction.replace(InstFlagCall, "").replace(callArgMatch.group(0), "")
        else:
            raise SyntaxError(f"function[{self.callObj}] params not found or not properly quoted")

    def _handleThoughtFlag(self):
        self.flag = InstFlag.Flag.THOUGHT
        self.instruction = self.instruction.replace(InstFlagThought, "").strip()

    def _handleDecomposeFlag(self):
        self.flag = InstFlag.Flag.DECOMPOSE
        self.instruction = self.instruction.replace(InstFlagDecompose, "").strip()

    def _extractPyCode(self):
        def _extractCode(text):
            pattern = r"```(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            return matches[0] if matches else None
        
        code = _extractCode(self.instruction)
        if code:
            lines = code.split('\n')
            minSpaces = min(len(line) - len(line.lstrip(' ')) for line in lines if line.strip())
            cleanedLines = [line[minSpaces:] for line in lines]
            return '\n'.join(cleanedLines)
        return ""

    def getOutputSyntaxes(self):
        return [struct.outputSyntax for struct in self.instOutput.outputStructs if struct.outputSyntax]
