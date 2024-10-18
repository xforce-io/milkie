from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.schema import NodeWithScore

@dataclass
class Response:

    respInt: Optional[int] = None
    respFloat: Optional[float] = None
    respBool: Optional[bool] = None
    respStr: Optional[str] = None
    respDict: Optional[Dict[str, Any]] = None
    respList: Optional[List[Any]] = None

    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None   

    @property
    def resp(self) -> Any:
        if self.respInt is not None:
            return self.respInt
        elif self.respFloat is not None:
            return self.respFloat
        elif self.respBool is not None:
            return self.respBool
        elif self.respStr is not None:
            return self.respStr
        elif self.respDict is not None:
            return self.respDict
        elif self.respList is not None:
            return self.respList
        else:
            return None

    def __str__(self) -> str:
        if self.respInt:
            return str(self.respInt)
        elif self.respFloat:
            return str(self.respFloat)
        elif self.respBool:
            return str(self.respBool)
        elif self.respStr:
            return self.respStr
        elif self.respList:
            return str(self.respList)
        elif self.respDict:
            return str(self.respDict)
        else:
            return ""