from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.schema import NodeWithScore

@dataclass
class Response:

    respStr: Optional[str] = None
    respDict: Optional[Dict[str, Any]] = None
    respList: Optional[List[Any]] = None
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None   

    @property
    def resp(self) -> Any:
        if self.respStr is not None:
            return self.respStr
        elif self.respDict is not None:
            return self.respDict
        elif self.respList is not None:
            return self.respList
        else:
            return None

    def __str__(self) -> str:
        if self.respStr:
            return self.respStr
        elif self.respDict:
            return str(self.respDict)
        elif self.respList:
            return str(self.respList)
        else:
            return str(self.respDict)