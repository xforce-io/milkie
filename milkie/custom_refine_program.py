import logging
from typing import Any, Type

from llama_index import BasePromptTemplate
from llama_index.types import BasePydanticProgram
from llama_index.bridge.pydantic import BaseModel
from llama_index.llm_predictor.base import LLMPredictorType
from llama_index.response_synthesizers.refine import StructuredRefineResponse

logger = logging.getLogger(__name__)


class CustomRefineProgram(BasePydanticProgram):

    def __init__(
            self, 
            prompt: BasePromptTemplate, 
            llm: LLMPredictorType, 
            output_cls: BaseModel
    ):
        self._prompt = prompt
        self._llm = llm
        self._output_cls = output_cls

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        import time
        t0 = time.time()
        if self._output_cls is not None:
            answer = self._llm.structured_predict(
                self._output_cls,
                self._prompt,
                **kwds,
            )
            answer = answer.json()
        else:
            answer = self._llm.predict(
                self._prompt, 
                **kwds,
            ).strip()
        t1 = time.time()
        qanswer = answer.replace("\n", "//")
        logger.debug(f"llm_call answer[{qanswer}] cost[{t1-t0}]")	
        return StructuredRefineResponse(answer=answer, query_satisfied=True)

    async def acall(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        if self._output_cls is not None:
            answer = await self._llm.astructured_predict(
                self._output_cls,
                self._prompt,
                **kwds,
            )
            answer = answer.json()
        else:
            answer = await self._llm.apredict(
                self._prompt,
                **kwds,
            )
        return StructuredRefineResponse(answer=answer, query_satisfied=True)
