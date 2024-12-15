import logging
from llama_index.core.prompts.base import PromptTemplate
from milkie.config.config import FRAMEWORK, EmbeddingConfig, SingleLLMConfig, LLMType

logger = logging.getLogger(__name__)

class ModelFactory:
    
    def __init__(self) -> None:
        self.llms = {}
        self.localLlm = None
        
        self.embedModel = {}

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
        
        if signatureModel == self.signatureLLMModel:
            return self.localLlm

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
            if config.framework == FRAMEWORK.VLLM:
                from milkie.llm.enhanced_vllm import EnhancedVLLM
                self.localLlm = EnhancedVLLM(
                    context_window=ctxLen,
                    concurrency=config.batchSize,
                    tensor_parallel_size=config.tensorParallelSize,
                    tokenizer_name=config.model,
                    model_name=config.model,
                    system_prompt=config.systemPrompt,
                    device=config.device,
                    port=config.port,
                    max_new_tokens=256,
                    tokenizer_kwargs=tokenizerArgs)
            elif config.framework == FRAMEWORK.LMDEPLOY:
                from milkie.llm.enhanced_lmdeploy import EnhancedLmDeploy
                self.localLlm = EnhancedLmDeploy(
                    context_window=ctxLen,
                    concurrency=config.batchSize,
                    tensor_parallel_size=config.tensorParallelSize,
                    tokenizer_name=config.model,
                    model_name=config.model,
                    system_prompt=config.systemPrompt,
                    device=config.device,
                    port=config.port,
                    max_new_tokens=256,
                    tokenizer_kwargs=tokenizerArgs)
            else :
                from milkie.llm.enhanced_hf_llm import EnhancedHFLLM
                self.localLlm = EnhancedHFLLM(
                    context_window=ctxLen,
                    concurrency=config.batchSize,
                    tensor_parallel_size=config.tensorParallelSize,
                    device=config.device,
                    port=config.port,
                    max_new_tokens=256,
                    model_kwargs=config.modelArgs.toJson(),
                    generate_kwargs=config.generationArgs.toJson(),
                    query_wrapper_prompt=PromptTemplate("{query}\n<|ASSISTANT|>\n"),
                    tokenizer_name=config.model,
                    model_name=config.model,
                    system_prompt=config.systemPrompt,
                    tokenizer_kwargs=tokenizerArgs,
                    is_chat_model=True,
                )

        logging.info(f"Building LLM with model[{config.model}] framework[{config.framework}] model_args[{repr(config.modelArgs)}] memory[{self.localLlm.getMem()}GB]")
        return self.localLlm
    
    def _getSignatureModel(self, config :SingleLLMConfig):
        if config.framework == FRAMEWORK.NONE:
            return config.name

        if config.modelArgs is None:
            return "%s-%s" % (config.model, config.framework)
        else:
            return "%s-%s-%s" % (config.model, config.framework, config.modelArgs.toJson())