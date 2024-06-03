from typing import List
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class ChunkAugment(BaseNodePostprocessor):
    def class_name(cls) -> str:
        return "ChunkAugment"

    def set_context(self, context):
        self.context = context

    def _postprocess_nodes(
            self, 
            nodes: List[NodeWithScore], 
            query_bundle: QueryBundle = None
    ) -> List[NodeWithScore]:
        newList = []
        for node in nodes:
            if node not in newList: 
                newList.append(node)

            nextNode = self.context.getGlobalMemory().getNextNode(node.node)
            if nextNode and node not in newList: 
                newList.append(node)
        return newList
            
        