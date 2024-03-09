from milkie.config.config import RerankConfig, RerankerType

from llama_index.postprocessor import FlagEmbeddingReranker


class Reranker:
    def __init__(self, rerankerConfig :RerankConfig):
        self.rerankerConfig = rerankerConfig
        if self.rerankerConfig.rerankerType == RerankerType.FLAGEMBED.name:
            self.reranker = FlagEmbeddingReranker(
                model=self.rerankerConfig.model,
                top_n=rerankerConfig.rerankTopK,
                use_fp16=True)
        else:
            self.reranker = None