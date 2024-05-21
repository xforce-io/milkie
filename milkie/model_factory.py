import logging
from llama_index.core.prompts.base import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from milkie.llm.enhanced_hf_llm import EnhancedHFLLM
from milkie.llm.enhanced_lmdeploy import EnhancedLmDeploy
from milkie.llm.enhanced_vllm import EnhancedVLLM
from milkie.config.config import FRAMEWORK, EmbeddingConfig, LLMConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    
    def __init__(self) -> None:
        self.llm = None
        self.signatureLLMModel = None
        self.embedModel = {}

    def getLLM(self, config :LLMConfig):
        signatureModel = self.__getSignatureModel(config)
        if signatureModel == self.signatureLLMModel:
            return self.llm

        llm = self.__setLLMModel(config)
        self.signatureLLMModel = signatureModel
        return llm

    def getEmbedding(self, config :EmbeddingConfig):
        logging.info(f"Building HuggingFaceEmbedding with model {config.model} from_cache[{config.model in self.embedModel}]")
        if config.model not in self.embedModel:
            self.embedModel[config.model] = HuggingFaceEmbedding(
                model_name=config.model,
                device=config.device)
        return self.embedModel[config.model]

    def __setLLMModel(self, config :LLMConfig):
        if self.llm is not None:
            del self.llm
            self.llm = None

        tokenizerArgs = {
            "max_length": config.ctxLen, 
            "use_fast": False, 
            "trust_remote_code": True,}

        if config.framework == FRAMEWORK.VLLM:
            self.llm = EnhancedVLLM(
                context_window=config.ctxLen,
                concurrency=config.batchSize,
                tokenizer_name=config.model,
                model_name=config.model,
                system_prompt=config.systemPrompt,
                device=config.device,
                max_new_tokens=256,
                tokenizer_kwargs=tokenizerArgs)
        elif config.framework == FRAMEWORK.LMDEPLOY:
            self.llm = EnhancedLmDeploy(
                context_window=config.ctxLen,
                concurrency=config.batchSize,
                tokenizer_name=config.model,
                model_name=config.model,
                system_prompt=config.systemPrompt,
                device=config.device,
                max_new_tokens=256,
                tokenizer_kwargs=tokenizerArgs)
        else :
            self.llm = EnhancedHFLLM(
                context_window=config.ctxLen,
                concurrency=config.batchSize,
                device=config.device,
                max_new_tokens=256,
                model_kwargs=config.modelArgs.toJson(),
                generate_kwargs=config.generationArgs.toJson(),
                query_wrapper_prompt=PromptTemplate("{query_str}\n<|ASSISTANT|>\n"),
                tokenizer_name=config.model,
                model_name=config.model,
                system_prompt=config.systemPrompt,
                tokenizer_kwargs=tokenizerArgs,
                is_chat_model=True,
            )

        logging.info(f"Building HuggingFaceLLM with model[{config.model}] framework[{config.framework}] model_args[{repr(config.modelArgs)}] memory[{self.llm.getMem()}GB]")
        return self.llm
    
    def __getSignatureModel(self, config :LLMConfig):
        return "%s-%s-%s" % (config.model, config.framework, config.modelArgs.toJson())