import logging
from typing import Any, Type

from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.types import BasePydanticProgram
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.response_synthesizers.refine import StructuredRefineResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CustomRefineProgram(BasePydanticProgram):

    def __init__(
            self, 
            prompt: BasePromptTemplate, 
            llm: LLMPredictorType, 
            **kwargs: Any) -> None:
        self._prompt = prompt
        self._llm = llm
        self._kwargs = kwargs

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        import time
        t0 = time.time()
        answer, _ = self._llm.predictBatch(
            prompt=self._prompt, 
            argsList=[kwds],
            **self._kwargs
        )[0]
        t1 = time.time()

        answer = answer.strip()
        qanswer = answer.replace("\n", "//")
        
        return StructuredRefineResponse(answer=answer, query_satisfied=True)

class CustomProgramFactory:

    def __init__(self, llm, **kwargs) -> None:
        self.llm = llm
        self.kwargs = kwargs

    def __call__(self, prompt: PromptTemplate) -> BasePydanticProgram:
        return CustomRefineProgram(
            prompt=prompt,
            llm=self.llm,
            kwargs=self.kwargs
        )