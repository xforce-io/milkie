from enum import Enum
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from milkie.agent.func_block.func_block import RepoFuncs
from milkie.agent.llm_block.step_llm_extractor import StepLLMExtractor
from milkie.config.constant import *
from milkie.context import VarDict
from milkie.settings import Settings
from milkie.types.object_type import ObjectType, ObjectTypeFactory
from milkie.utils.data_utils import codeToLines, extractBlock, extractJsonBlock, isBlock, unescape
from milkie.vm.vm import VM

logger = logging.getLogger(__name__)

class OutputSyntaxFormat(Enum):
    NORMAL = 1
    REGEX = 2
    EXTRACT = 3
    CHECK = 4
    JSON = 5
    JSONLIST = 6
    OBJECT = 7

class ResultOutputProcessSingle:
    def __init__(self, storeVar: str, output: Any=None, errmsg: str=None):
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
        
    def addSuccess(self, storeVar: Optional[str], output: Any):
        if storeVar is None:
            return 
        
        self._results[storeVar] = ResultOutputProcessSingle(storeVar, output, None)

    def addError(self, storeVar: Optional[str], errmsg: str):
        if storeVar is None:
            return 
        
        if not errmsg:
            errmsg = "unknown error"
        self._results[storeVar] = ResultOutputProcessSingle(storeVar, None, errmsg)

    def isSuccess(self, storeVar: str):
        return storeVar in self._results and not self._results[storeVar].hasError()

    def hasError(self) -> bool:
        return any(result.hasError() for result in self._results.values())

    def getErrMsgs(self) -> List[str]:
        return [self._results[storeVar].errmsg for storeVar in self._results if self._results[storeVar].hasError()]

    def setStoreVarDict(self, varDict: VarDict, contextLen: int=None):
        for storeVar, result in self._results.items():
            if result.hasError():
                continue

            if type(result.output) == str and isBlock("json", result.output):
                jsonStr = extractBlock("json", result.output).replace("\n", '')
                if jsonStr.startswith("{{") and jsonStr.endswith("}}"):
                    jsonStr = jsonStr[1:-1]

                try:
                    varDict.setGlobal(storeVar, json.loads(jsonStr))
                except Exception as e:
                    logger.error(f"Error parsing JSON: {e}")
                    varDict.setGlobal(storeVar, jsonStr)
            else:
                if type(result.output) == str and \
                        contextLen is not None and \
                        len(result.output) > contextLen:
                    result.output = result.output[:contextLen]
                    logger.warning(f"output[{result.output}] is too long, actualLen={len(result.output)}, contextLen={contextLen}")
                varDict.setGlobal(
                    storeVar, 
                    result.output.strip() if type(result.output) == str else result.output)
    
    def __str__(self) -> str:
        return f"ResultOutputProcess(results={self._results})"

class OutputSyntax:
    def __init__(
            self, 
            syntax: str,
            globalObjectTypes: ObjectTypeFactory):
        self.originalSyntax = syntax.strip()
        self.globalObjectTypes = globalObjectTypes
        
        self.format = self._determineFormat()
        self.regExpr = None
        self.extractPattern = None
        self.checkPattern = None
        self.errorMessage = None
        self.objectTypes = []
        self._parse()
    
    def getOriginalSyntax(self):
        return self.originalSyntax

    def _determineFormat(self) -> OutputSyntaxFormat:
        if self.originalSyntax.startswith("r'") or self.originalSyntax.startswith('r"'):
            return OutputSyntaxFormat.REGEX
        elif self.originalSyntax.startswith("e'") or self.originalSyntax.startswith('e"'):
            return OutputSyntaxFormat.EXTRACT
        elif self.originalSyntax.startswith("o'") or self.originalSyntax.startswith('o"'):
            return OutputSyntaxFormat.OBJECT
        elif self.originalSyntax.startswith("c```"):
            return OutputSyntaxFormat.CHECK
        elif self.originalSyntax.strip() == "json":
            return OutputSyntaxFormat.JSON
        elif self.originalSyntax.strip() == "jsonl":
            return OutputSyntaxFormat.JSONLIST
        else:
            return OutputSyntaxFormat.NORMAL

    def _parse(self):
        if self.format == OutputSyntaxFormat.NORMAL or \
                self.format == OutputSyntaxFormat.JSON or \
                self.format == OutputSyntaxFormat.JSONLIST:
            return

        content = self.originalSyntax[1:]
        if self.format == OutputSyntaxFormat.CHECK:
            quoteStr = content[:3]
        else:
            quoteStr = content[0]
            
        lastQuoteIndex = -1
        i = 1
        while i < len(content):
            if content[i:i+len(quoteStr)] == quoteStr and content[i-1] != '\\':
                lastQuoteIndex = i
            i += 1

        if lastQuoteIndex == -1:
            raise ValueError("Invalid syntax: missing closing quote")

        pattern = content[len(quoteStr):lastQuoteIndex]
        remaining = content[lastQuoteIndex+len(quoteStr):].strip()

        if remaining.startswith('/'):
            self.errorMessage = remaining[1:].strip().strip('"\'')
        else:
            self.errorMessage = None

        if self.format == OutputSyntaxFormat.REGEX:
            self.regExpr = re.compile(pattern)
        elif self.format == OutputSyntaxFormat.EXTRACT:
            self.extractPattern = pattern
        elif self.format == OutputSyntaxFormat.CHECK:
            self.checkPattern = pattern
        elif self.format == OutputSyntaxFormat.OBJECT:
            titles = [title.strip() for title in pattern.split(',')]
            if len(titles) == 0:
                raise ValueError(f"Invalid syntax: missing object type[{pattern}]")

            self.objectTypes = self.globalObjectTypes.getTypes(titles)
        else:
            raise ValueError(f"Invalid format: {self.format}")

    def getOutputSyntax(self):
        return unescape(self.originalSyntax)

    def getObjectOutputSyntax(self):
        return self.objectTypes

    def processOutput(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: Any,
            varDict: VarDict,
            vm: VM) -> Any:
        if type(output) == str:
            if self.format == OutputSyntaxFormat.REGEX:
                return self.regExpr.search(output).group(1)
            elif self.format == OutputSyntaxFormat.EXTRACT:
                allArgs = varDict.getAllDict()
                allArgs["toExtract"] = self.extractPattern
                allArgs["text"] = output
                return stepLLMExtractor.completionAndFormat(
                    args=allArgs)
            elif self.format == OutputSyntaxFormat.JSON or \
                    self.format == OutputSyntaxFormat.JSONLIST or \
                    self.format == OutputSyntaxFormat.OBJECT:
                return extractJsonBlock(output)

        if self.format == OutputSyntaxFormat.CHECK:
            try:
                return vm.execPython(
                    code=self.checkPattern, 
                    varDict=varDict.getAllDict())
            except Exception as e:
                logger.warning(f"Error running code: {e}")
                return None
        return output

    def __str__(self) -> str:
        return f"OutputSyntax(originalSyntax={self.originalSyntax}, format={self.format}, regExpr={self.regExpr}, extractPattern={self.extractPattern}, errorMessage={self.errorMessage})"

class OutputStruct:
    def __init__(self, outputSyntax: OutputSyntax = None, storeVar: str = None):
        self.outputSyntax = outputSyntax
        self.storeVar = storeVar

    def processOutput(
            self, 
            stepLLMExtractor: StepLLMExtractor, 
            output: Any,
            varDict: VarDict,
            vm: VM) -> Any:
        return self.outputSyntax.processOutput(
            stepLLMExtractor=stepLLMExtractor, 
            output=output, 
            varDict=varDict,
            vm=vm)

    def getOutputSyntaxFormat(self):
        return self.outputSyntax.format if self.outputSyntax else None

    def __str__(self) -> str:
        return f"OutputStruct(outputSyntax={self.outputSyntax}, storeVar={self.storeVar})"

class InstrOutput:
    def __init__(self):
        self.outputStructs: List[OutputStruct] = []
        self.reset()

    def reset(self):
        self._processed = False
        self._currentResult: ResultOutputProcess = None

    def addOutputStruct(self, outputStruct: OutputStruct):
        self.outputStructs.append(outputStruct)
    
    def getCertainOutputSyntax(self, format: OutputSyntaxFormat) -> str:
        for outputStruct in self.outputStructs:
            if outputStruct.outputSyntax and outputStruct.outputSyntax.format == format:
                return outputStruct.outputSyntax.getOutputSyntax()
        return None
    
    def getCurrentResult(self):
        return self._currentResult

    def storeResultToVarDict(self, varDict: VarDict, contextLen: int):
        if self._currentResult:
            self._currentResult.setStoreVarDict(varDict, contextLen)
    
    def processOutputAndStore(
            self, 
            output: Any,
            stepLLMExtractor: StepLLMExtractor, 
            varDict: VarDict,
            vm: VM,
            retry: bool,
            contextLen: int):
        if self._processed:
            return

        self._processOutput(
            output=output,
            stepLLMExtractor=stepLLMExtractor, 
            varDict=varDict,
            vm=vm)
        if not self._currentResult.hasError():
            self._processed = True
            self.storeResultToVarDict(varDict, contextLen)
        else:
            if not retry:
                raise RuntimeError(f"fail process output[{output}]")
    
    def isProcessed(self):
        return self._processed

    def hasError(self):
        return self._currentResult.hasError()

    def getErrMsgs(self):
        return self._currentResult.getErrMsgs()

    def len(self):
        return len(self.outputStructs)

    def _processOutput(
            self, 
            output: Any,
            stepLLMExtractor: StepLLMExtractor, 
            varDict: VarDict,
            vm: VM):
        if self._currentResult is None:
            self._currentResult = ResultOutputProcess()

        for outputStruct in self.outputStructs:
            if self._currentResult.isSuccess(outputStruct.storeVar):
                continue

            if outputStruct.outputSyntax:
                processedOutput = outputStruct.processOutput(
                    output=output, 
                    stepLLMExtractor=stepLLMExtractor, 
                    varDict=varDict,
                    vm=vm)
                if processedOutput is None or \
                        (type(processedOutput) == str and processedOutput.strip() == ExprNoInfoToExtract):
                    self._currentResult.addError(
                        storeVar=outputStruct.storeVar,
                        errmsg=outputStruct.outputSyntax.errorMessage)
                else:
                    if outputStruct.getOutputSyntaxFormat() != OutputSyntaxFormat.CHECK:
                        output = processedOutput
                        
                    self._currentResult.addSuccess(
                        storeVar=outputStruct.storeVar,
                        output=output)
            else:
                self._currentResult.addSuccess(
                    storeVar=outputStruct.storeVar,
                    output=output)

class SyntaxParser:
    class Flag(Enum):
        NONE = 1
        RET = 2
        CODE = 3
        IF = 4
        GOTO = 5
        PY = 6
        THOUGHT = 7

    class TypeCall(Enum):
        STDIN = 1
        Agent = 2

    def __init__(
            self, 
            settings: Settings,
            label: str,
            instruction: str,
            repoFuncs: RepoFuncs,
            globalObjectTypes: ObjectTypeFactory) -> None:
        self.settings = settings

        self.flag = SyntaxParser.Flag.NONE
        self.label = label
        self.model = None
        self.instOutput = InstrOutput()
        self.returnVal = False 
        self.instruction = instruction.strip()
        self.funcsToCall = []
        
        self.repoFuncs = repoFuncs
        self.globalObjectTypes = globalObjectTypes

        self.typeCall = None
        self.callObj = None
        self.callArg = None

        self.parseInstruction()

    def reset(self):
        self.instOutput.reset()

    def getInstruction(self):
        return self.instruction

    def getInstrOutput(self):
        return self.instOutput

    def getNormalFormat(self) -> str:
        return self.instOutput.getCertainOutputSyntax(OutputSyntaxFormat.NORMAL)

    def getJsonFormat(self) -> str:
        return self.instOutput.getCertainOutputSyntax(OutputSyntaxFormat.JSON)

    def getJsonListFormat(self) -> str:
        return self.instOutput.getCertainOutputSyntax(OutputSyntaxFormat.JSONLIST)

    def getStoreVars(self):
        return [struct.storeVar for struct in self.instOutput.outputStructs if struct.storeVar]

    def getNormalOutputSyntax(self):
        outputStructs = [struct for struct in self.instOutput.outputStructs if struct.outputSyntax and struct.outputSyntax.format == OutputSyntaxFormat.NORMAL]
        if not outputStructs:
            return None
        return outputStructs[0].outputSyntax.getOriginalSyntax()

    def getObjectOutputSyntax(self) -> List[ObjectType]:
        outputStructs = [struct for struct in self.instOutput.outputStructs if struct.outputSyntax and struct.outputSyntax.format == OutputSyntaxFormat.OBJECT]
        if len(outputStructs) > 1:
            raise Exception("Multiple object types found in instruction with output syntax")
        elif len(outputStructs) == 1:
            return outputStructs[0].outputSyntax.getObjectOutputSyntax()
        else:
            return None

    def parseInstruction(self):
        self._handleOutputAndStore()
        self._handleFlags()
        self._handleModel() # model should be handled after flags
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
                outputSyntax = OutputSyntax(content, self.globalObjectTypes) if content else None
                lastSeparator = separator
            elif separator == "->":
                if lastSeparator == "->" and outputSyntax is None:
                    raise Exception("Invalid syntax: consecutive store variables without output syntax")
                storeVar = content
                self.instOutput.addOutputStruct(OutputStruct(outputSyntax=outputSyntax, storeVar=storeVar))
                outputSyntax = None
                lastSeparator = separator
        
        self.instOutput.addOutputStruct(OutputStruct(outputSyntax=outputSyntax, storeVar=self.label))

        # Check for invalid syntax
        if '->' in self.instruction:
            raise Exception("Invalid syntax: store variable in instruction")

        self._checkOutputSyntaxesConflicts()

    def _handleModel(self):
        if self.flag == SyntaxParser.Flag.CODE:
            self.model = self.settings.getLLMCode()
            return

        for name in self.settings.getAllLLMs():
            if f"[{name}]" in self.instruction:
                self.model = self.settings.getLLM(name)
                self.instruction = self.instruction.replace(f"[{name}]", "")
                return
        self.model = self.settings.getLLMDefault()

    def _handleFunctions(self):
        if self.repoFuncs is not None:
            for funcName, funcBlock in self.repoFuncs.getAll().items():
                pattern = funcBlock.getFuncNamePattern()
                if pattern not in self.instruction:
                    continue
                
                params, funcCallPattern = self._parseFuncParams(pattern, self.instruction)
                if params is not None:
                    newFuncBlock = funcBlock.createFuncCall()
                    newFuncBlock.setParams(params)
                    newFuncBlock.setFuncCallPattern(funcCallPattern)
                    newFuncBlock.compile()
                    self.funcsToCall.append(newFuncBlock)
                else:
                    raise SyntaxError(f"function[{funcName}] params not found")

    def _handleFlags(self):
        flagHandlers = {
            InstFlagRet: self._handleRetFlag,
            InstFlagCode: self._handleCodeFlag,
            InstFlagIf: self._handleIfFlag,
            InstFlagGoto: self._handleGotoFlag,
            InstFlagPy: self._handlePyFlag,
        }

        flagsFound = [flag for flag in flagHandlers.keys() if flag in self.instruction]

        if len(flagsFound) > 1:
            raise Exception("Multiple flags found in instruction")

        if flagsFound:
            flagHandlers[flagsFound[0]]()
        else:
            self.flag = SyntaxParser.Flag.NONE

    def _handleRetFlag(self):
        self.flag = SyntaxParser.Flag.RET
        self.returnVal = self.instruction.startswith(InstFlagRet)
        self.instruction = self.instruction.replace(InstFlagRet, "").strip()

    def _handleCodeFlag(self):
        self.flag = SyntaxParser.Flag.CODE
        self.instruction = self.instruction.replace(InstFlagCode, "")

    def _handleIfFlag(self):
        self.flag = SyntaxParser.Flag.IF

    def _handleGotoFlag(self):
        self.flag = SyntaxParser.Flag.GOTO
        gotoParts = self.instruction.split(InstFlagGoto)
        self.label = gotoParts[1].split()[0].strip()
        self.instruction = gotoParts[0].strip() + " ".join(gotoParts[1].split()[1:]).strip()

    def _handlePyFlag(self):
        self.flag = SyntaxParser.Flag.PY
        pyCode = self._extractPyCode()
        self.instruction = pyCode if pyCode else self.instruction.replace(InstFlagPy, "").strip()

    def _extractPyCode(self):
        def _extractCode(text):
            pattern = r"```(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            return matches[0] if matches else None
        
        code = _extractCode(self.instruction)
        if code:
            lines = codeToLines(code)
            minSpaces = min(len(line) - len(line.lstrip(' ')) for line in lines if line.strip())
            cleanedLines = [line[minSpaces:] for line in lines]
            return '\n'.join(cleanedLines)
        return ""

    def getOutputSyntaxes(self):
        return [struct.outputSyntax for struct in self.instOutput.outputStructs if struct.outputSyntax]

    def _checkOutputSyntaxesConflicts(self):
        outputSyntaxes = self.getOutputSyntaxes()
        for outputSyntax in outputSyntaxes:
            if outputSyntax.format == OutputSyntaxFormat.OBJECT:
                if len(outputSyntaxes) > 1:
                    raise Exception("Multiple object types found in instruction with output syntax")

    def _parseFuncParams(self, pattern: str, instruction: str) -> Tuple[List[str], str]:
        # 查找函数调用的起始位置
        pattern_pos = instruction.find(pattern)
        if pattern_pos == -1:
            return None, None
        
        # 查找参数列表开始的左括号
        open_paren_pos = instruction.find('(', pattern_pos + len(pattern))
        if open_paren_pos == -1:
            return None, None
        
        # 手动寻找与左括号匹配的右括号，考虑嵌套括号和字符串字面量
        bracket_count = 1
        in_string = False
        string_char = None
        pos = open_paren_pos + 1
        
        while pos < len(instruction) and bracket_count > 0:
            char = instruction[pos]
            
            # 处理字符串字面量开始/结束
            if not in_string and char in ['"', "'"]:
                # 检查三引号
                if pos + 2 < len(instruction) and instruction[pos:pos+3] in ['"""', "'''"]:
                    in_string = True
                    string_char = instruction[pos:pos+3]
                    pos += 3
                    continue
                else:
                    in_string = True
                    string_char = char
                    pos += 1
                    continue
            elif in_string:
                # 检查是否到达字符串结束
                if (len(string_char) == 3 and pos + 2 < len(instruction) and 
                        instruction[pos:pos+3] == string_char) or \
                   (len(string_char) == 1 and char == string_char and 
                        (pos == 0 or instruction[pos-1] != '\\')):
                    in_string = False
                    if len(string_char) == 3:
                        pos += 3
                        continue
                pos += 1
                continue
            
            # 只有不在字符串内部时才计数括号
            if char == '(':
                bracket_count += 1
            elif char == ')':
                bracket_count -= 1
            
            pos += 1
        
        if bracket_count != 0:
            return None, None  # 没有找到匹配的右括号
        
        close_paren_pos = pos - 1
        
        # 提取参数字符串和完整的函数调用
        params_str = instruction[open_paren_pos + 1:close_paren_pos].strip()
        func_call_pattern = instruction[pattern_pos:close_paren_pos + 1]
        
        if not params_str:
            return [], func_call_pattern
            
        # 以下保持原来的参数分割和处理逻辑不变
        # 解析带引号的参数
        params = []
        i = 0
        current_param = []
        in_quotes = False
        quote_type = None
        
        while i < len(params_str):
            char = params_str[i]
            
            # 检查引号开始/结束
            if char in ['"', "'"]:
                # 检查三引号
                if i + 2 < len(params_str) and params_str[i:i+3] in ['"""', "'''"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_type = params_str[i:i+3]
                        i += 3
                        continue
                    elif quote_type == params_str[i:i+3]:
                        in_quotes = False
                        quote_type = None
                        i += 3
                        continue
                # 单引号或双引号
                elif not in_quotes:
                    in_quotes = True
                    quote_type = char
                    i += 1
                    continue
                elif quote_type == char:
                    in_quotes = False
                    quote_type = None
                    i += 1
                    continue
            
            # 处理参数分隔符
            if char == ',' and not in_quotes:
                param = ''.join(current_param).strip()
                if param:
                    params.append(param)
                current_param = []
                i += 1
                continue
                
            current_param.append(char)
            i += 1
            
        # 添加最后一个参数
        if current_param:
            param = ''.join(current_param).strip()
            if param:
                params.append(param)
                
        # 去除参数外层的引号
        cleaned_params = []
        for param in params:
            # 去除三引号
            if (param.startswith('"""') and param.endswith('"""')) or \
               (param.startswith("'''") and param.endswith("'''")):
                param = param[3:-3]
            # 去除单引号或双引号
            elif (param.startswith('"') and param.endswith('"')) or \
                 (param.startswith("'") and param.endswith("'")):
                param = param[1:-1]
            cleaned_params.append(param)
            
        return cleaned_params, func_call_pattern
