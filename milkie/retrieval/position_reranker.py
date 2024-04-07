from typing import List
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
from llama_index.legacy.schema import NodeWithScore, QueryBundle

class PositionReranker(BaseNodePostprocessor):

    def class_name(cls) -> str:
        return "PositionReranker"

    def _postprocess_nodes(
            self, 
            nodes: List[NodeWithScore], 
            query_bundle: QueryBundle = None
    ) -> List[NodeWithScore]:
        return sorted(nodes, key=lambda x: x.node.start_char_idx if x.node.start_char_idx else 0)