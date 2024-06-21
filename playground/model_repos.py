import re

class Model:
    def __init__(
            self, 
            modelSeries, 
            rootpath, 
            modelpath,
            ftFlag=None,
            tensorParrallelSize=1,
            ) -> None:
        self.modelSeries = modelSeries
        self.rootpath = rootpath
        self.modelpath = modelpath
        self.ftFlag = ftFlag
        self.tensorParrallelSize = tensorParrallelSize

        self.isChat = None
        self.size = None
        self.quantMethod = None
        self.quantBits = None
        self._parseAttributesFromModepath()

        if self.isChat is None:
            raise ValueError(f"Unknown model type: {self.modelpath}")
        
        if self.size is None:
            raise ValueError(f"Unknown model size: {self.modelpath}")

        self.name = f"{self.modelSeries.lower()}"
        if self.ftFlag != None:
            self.name += f"-{self.ftFlag}"
        if self.isChat:
            self.name += "-chat"
        self.name += f"-{self.size}b"
        if self.quantMethod:
            self.name += f"-{self.quantMethod}"
            if self.quantBits:
                self.name += f"-int{self.quantBits}"

    def getModelName(self) -> str:
        return self.name

    def getModelPath(self) -> str:
        return f"{self.rootpath}{self.modelpath}"

    def getTensorParrallelSize(self) -> int:
        return self.tensorParrallelSize

    def _parseAttributesFromModepath(self) -> None:
        self.isChat = "chat" in self.modelpath.lower()

        tmp = re.search(r"\d+b", self.modelpath.lower())
        if tmp:
            self.size = int(tmp.group(0)[:-1])
        else:
            raise ValueError(f"Unknown model size: {self.modelpath}")

        if "gptq" in self.modelpath.lower():
            self.quantMethod = "gptq"
            self.quantBits = 8
        elif "awq" in self.modelpath.lower():
            self.quantMethod = "awq"

        tmp = re.search(r"int\d", self.modelpath.lower())
        if tmp:
            if self.quantMethod == "gptq":
                self.quantBits = int(tmp.group(0)[-1:])
                if self.quantBits != 4 and self.quantBits != 8:
                    raise ValueError(f"Unknown quant bits: {self.modelpath}")
            else:
                raise ValueError(f"Unknown model size: {self.modelpath}")


class ModelRepo:
    def __init__(self) -> None:
        self.models = {}
    
    def addModel(self, model :Model) -> None:
        if model.getModelName() in self.models:
            raise ValueError(f"Model {model.getModelName()} already exists")
        
        self.models[model.getModelName()] = model
    
    def getModel(self, name :str) -> Model:
        if name not in self.models:
            raise ValueError(f"Model {name} not found")

        return self.models[name]

GModelRepo = ModelRepo()

Rootpath = "/mnt/data1/.cache/modelscope/hub/"
GModelRepo.addModel(Model("qwenv1.5", Rootpath, "qwen/Qwen1.5-14B-Chat/"))
GModelRepo.addModel(Model("qwenv1.5", Rootpath, "qwen/Qwen1___5-14B-Chat-GPTQ-Int4/"))
GModelRepo.addModel(Model("qwenv1.5", Rootpath, "qwen/Qwen1___5-14B-Chat-GPTQ-Int8/"))
GModelRepo.addModel(Model("qwenv1.5", Rootpath, "qwen/Qwen1___5-14B-Chat-AWQ/"))
GModelRepo.addModel(Model("qwenv2", Rootpath, "qwen/Qwen2-57B-A14B/"))

RootpathAishuReader = "/mnt/data2/.cache/huggingface/hub/"
GModelRepo.addModel(Model("aishuv2", RootpathAishuReader, "Qwen-14B-Chat-1.5-aishuV2"))
GModelRepo.addModel(Model("aishuv2", RootpathAishuReader, "Qwen-14B-Chat-1.5-aishuV2-awq"))
GModelRepo.addModel(Model("aishuv2", RootpathAishuReader, "Qwen-14B-Chat-1.5-aishuV2-gptq-int8"))

RootpathInternlm2 = "/mnt/data3/models/"
GModelRepo.addModel(Model("internlmv2", RootpathInternlm2, "internlm2-chat-20b_v2"))
GModelRepo.addModel(Model("internlmv2", RootpathInternlm2, "../trained_models/internlm2-chat-20b_log_3k5_ruozhiba_longrag", ftFlag="longrag"))

RootpathYi = "/mnt/data3/models/01ai/"
GModelRepo.addModel(Model("Yi", RootpathYi, "Yi-1___5-34B-Chat/"))

GModelRepo.addModel(Model("deepseekv2", None, "deepseek-236B-chat"))

if __name__ == "__main__" :
    for name, model in GModelRepo.models.items():
        print(f"ModelName: {name}")
