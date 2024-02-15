from typing import List
from llama_index import Response, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.llms import AzureOpenAI
import torch
from milkie.config.config import EmbeddingConfig, GlobalConfig, LLMConfig, LLMType, MemoryTermConfig

from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from milkie.index.index import Index

from milkie.memory.memory import Memory

SystemPromptCn = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM是由StabilityAI开发的强大的大语言模型。
- StableLM对于能够帮助用户很兴奋,并且拒绝做任何可能对用户有害处的事情。
- StableLM不只是一个信息源，它能够写诗、写故事和讲笑话。
- StableLM拒绝做任何会伤害人类的事情。
- 用中文进行回答
"""

class Context:
    def __init__(self, globalConfig :GlobalConfig) -> None:
        self.config :GlobalConfig = globalConfig

        self.curQuery :str = None
        self.retrievalResult = None
        self.decisionResult :Response = None
        self.engine = None
        self.instructions = []

        self.__buildLLM(self.config.llmConfig)
        self.__buildEmbedding(self.config.embeddingConfig)
        self.__buildMemory(self.config.memoryConfig)
        self.__buildIndex(self.config.indexConfig)
        
    def setCurQuery(self, query):
        self.curQuery = query

    def getCurQuery(self):
        return self.curQuery

    def getCurInstruction(self):
        return None if len(self.instructions) == 0 else self.instructions[-1]

    def setRetrievalResult(self, retrievalResult):
        self.retrievalResult = retrievalResult

    def setDecisionResult(self, decisionResult :Response):
        self.decisionResult = decisionResult

    def __buildLLM(self, config :LLMConfig):
        if config.type == LLMType.HUGGINGFACE:
            self.llm = HuggingFaceLLM(
                context_window=config.ctxLen,
                max_new_tokens=256,
                model_kwargs={"torch_dtype":torch.bfloat16, "trust_remote_code" :True},
                generate_kwargs={"temperature": 0, "do_sample": False},
                system_prompt=SystemPromptCn,
                query_wrapper_prompt=PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>"),
                tokenizer_name=config.model,
                model_name=config.model,
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": config.ctxLen, "use_fast": False, "trust_remote_code": True},
            )
        elif config.type == LLMType.AZURE_OPENAI:
            self.llm = AzureOpenAI(
                azure_endpoint=config.azureEndpoint,
                azure_deployment=config.deploymentName,
                api_version=config.apiVersion,
                api_key=config.apiKey,
                system_prompt=SystemPromptCn,
                temperature=0)

    def __buildEmbedding(self, config :EmbeddingConfig):
        self.embedding = HuggingFaceEmbedding(
            model_name=config.model,
            device=config.device)

    def __buildMemory(self, config :List[MemoryTermConfig]):
        self.serviceContext = ServiceContext.from_defaults(
            embed_model=self.embedding,
            chunk_size=self.config.indexConfig.chunkSize,
            chunk_overlap=self.config.indexConfig.chunkOverlap,
            llm=self.llm)

        self.memory = Memory(config, self.serviceContext)

    def __buildIndex(self, indexConfig):
        denseIndex = VectorStoreIndex(
            self.memory.nodes,
            storage_context=self.memory.storageContext,
            service_context=self.serviceContext)
        
        self.index = Index(denseIndex)