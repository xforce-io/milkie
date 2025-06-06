import logging
from milkie.cache.cache_kv import CacheKVMgr
from milkie.config.config import FRAMEWORK, EmbeddingConfig, SingleLLMConfig, LLMType

logger = logging.getLogger(__name__)

class ModelFactory:
    
    def __init__(self) -> None:
        self.llms = {}
        self.localLlm = None
        
        self.embedModel = {}
        self._cacheMgr = CacheKVMgr("data/cache/", category='model', expireTimeByDay=2)

    def getLLM(
            self, 
            config :SingleLLMConfig):
        signatureModel = self._getSignatureModel(config)
        if config.type == LLMType.GEN_OPENAI:
            if signatureModel in self.llms:
                return self.llms[signatureModel]
            else:
                llm = self._setLLMModel(config)
                self.llms[signatureModel] = llm
                return llm
        
        self.localLlm = self._setLLMModel(config)
        return self.localLlm

    def getEmbedding(self, config :EmbeddingConfig):
        logging.info(f"Building HuggingFaceEmbedding with model {config.model} from_cache[{config.model in self.embedModel}]")
        if config.model not in self.embedModel:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.embedModel[config.model] = HuggingFaceEmbedding(
                model_name=config.model,
                device=config.device)
        return self.embedModel[config.model]

    def _setLLMModel(
            self, 
            config :SingleLLMConfig):
        if self.localLlm is not None:
            del self.localLlm
            self.localLlm = None

        ctxLen = config.ctxLen

        tokenizerArgs = {
            "max_length": ctxLen, 
            "use_fast": False, 
            "trust_remote_code": True,}

        if config.type == LLMType.GEN_OPENAI:
            from milkie.llm.enhanced_openai import EnhancedOpenAI
            self.localLlm = EnhancedOpenAI(
                cacheMgr=self._cacheMgr,
                model_name=config.model,
                system_prompt=config.systemPrompt,
                endpoint=config.endpoint,
                api_key=config.apiKey,
                context_window=ctxLen,
                concurrency=config.batchSize,
                tensor_parallel_size=config.tensorParallelSize,
                tokenizer_name=None,
                device=config.device,
                port=config.port,
                tokenizer_kwargs=tokenizerArgs)
        else:
            raise Exception(f"Unsupported LLM type: {config.type}")

        logging.info(f"Building LLM with model[{config.model}] framework[{config.framework}] model_args[{repr(config.modelArgs)}] memory[{self.localLlm.getMem()}GB]")
        return self.localLlm
    
    def _getSignatureModel(self, config :SingleLLMConfig):
        if config.framework == FRAMEWORK.NONE:
            return config.name

        if config.modelArgs is None:
            return "%s-%s" % (config.model, config.framework)
        else:
            return "%s-%s-%s" % (config.model, config.framework, config.modelArgs.toJson())