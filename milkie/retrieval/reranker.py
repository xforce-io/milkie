from milkie.config.config import RerankConfig, RerankerType

from llama_index.postprocessor import FlagEmbeddingReranker


class Reranker:
    def __init__(self, rerankerConfig :RerankConfig):
        self.rerankerConfig = rerankerConfig
        if self.rerankerConfig.rerankerType == RerankerType.FLAGEMBED.name:
            self.reranker = FlagEmbeddingReranker(
                top_n=rerankerConfig.rerankTopK,
                use_fp16=True)
        else:
            raise Exception(f"Not supported reranker type[{self.rerankerConfig.rerankerType}]")