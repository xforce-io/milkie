from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from openai.types.chat.chat_completion_message import ChatCompletionMessage

from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import ChatResponse

from milkie.config.constant import KeywordEnd


@dataclass
class Response:

    respInt: Optional[int] = None
    respFloat: Optional[float] = None
    respBool: Optional[bool] = None
    respStr: Optional[str] = None
    respDict: Optional[Dict[str, Any]] = None
    respList: Optional[List[Any]] = None
    respGen: Optional[Generator[ChatResponse, None, None]] = None

    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None   

    @staticmethod
    def buildFrom(resp: Any) -> "Response":
        if isinstance(resp, str):
            return Response(respStr=resp)
        elif isinstance(resp, int):
            return Response(respInt=resp)
        elif isinstance(resp, float):
            return Response(respFloat=resp)
        elif isinstance(resp, bool):
            return Response(respBool=resp)
        elif isinstance(resp, dict):
            return Response(respDict=resp)
        elif isinstance(resp, list):
            return Response(respList=resp)
        elif isinstance(resp, Generator):
            return Response(respGen=resp)
        else:
            raise ValueError(f"Unsupported response type: {type(resp)}")

    def getChoice0Message(self) -> ChatCompletionMessage:
        return self.metadata["chatCompletion"].choices[0].message
    
    def getChoicesMessages(self) -> List[ChatCompletionMessage]:
        return [choice.message for choice in self.metadata["chatCompletion"].choices]

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
        elif self.respGen is not None:
            return self.respGen
        else:
            return None

    def isEnd(self) -> bool:
        return KeywordEnd in self.respStr

    @staticmethod
    def isNaivePyType(resp: Any) -> bool:
        return isinstance(resp, str) or \
            isinstance(resp, int) or \
            isinstance(resp, float) or \
            isinstance(resp, bool) or \
            isinstance(resp, list) or \
            isinstance(resp, dict)

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
