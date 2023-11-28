from ..tools.all_tools import kTools

class PromptMaker(object):

    kSep = "-------------\n"
    
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.prompt = ""

    def injectInstruction(self, instruction):
        self.prompt += PromptMaker.kSep + "Instruction: " + instruction + "\n"

    def injectTools(self) :
        self.prompt += PromptMaker.kSep + "Tools available as below\n"
        for tool in kTools:
            self.prompt += PromptMaker.kSep + tool.describe()
    
    def injectFormat(self):
        self.prompt += PromptMaker.kSep + "Response Format: \n"'''
        ToolName: toolname 
        Arg1: arg1
        ...
        ArgN: argN
        ''' + PromptMaker.kSep
        
    def isSepLine(self, line):
        return line == PromptMaker.kSep.strip()
        
    def formatParser(self, resp):
        #parse the response according to the format, return a dict
        # {
        #     "errno": Errno,
        #     "toolName": "toolname",
        #     "Arg1": "arg1",
        #     ...
        #     "ArgN": "argN"
        # }
        def extractArg(line, prefix):
            items = line.strip().split(":")
            if len(items) == 1 or items[0].strip() != prefix:
                return None
            return items[1].strip()
        
        lines = resp.split("\n")
        toolName = None
        args = []
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            if not self.isSepLine(lines[i]):
                continue
            
            if i+1 < len(lines):
                toolName = extractArg(lines[i+1], "ToolName")
            else:
                break

            j = i+2
            while j < len(lines):
                arg = extractArg(lines[j], "Arg")
                if arg is None:
                    break

                args.append(arg)
                j += 1
        
        if toolName is None:
            return None        
        
        return {
            "toolName": toolName,
            "args": args
        }

    def makePrompt(self, instruction):
        self.reset()
        self.injectInstruction(instruction)
        self.injectTools()
        self.injectFormat()
        return self.prompt