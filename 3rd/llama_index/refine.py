import logging, time
from typing import Any, Callable, Generator, Optional, Sequence, Type, cast

from llama_index.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.indices.utils import truncate_text
from llama_index.llm_predictor.base import LLMPredictorType
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.prompts.mixin import PromptDictType
from llama_index.response.utils import get_response_text
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.service_context import ServiceContext
from llama_index.types import RESPONSE_TEXT_TYPE, BasePydanticProgram

logger = logging.getLogger(__name__)


class StructuredRefineResponse(BaseModel):
    """
    Used to answer a given query based on the provided context.

    Also indicates if the query was satisfied with the provided answer.
    """

    answer: str = Field(
        description="The answer for the given query, based on the context and not "
        "prior knowledge."
    )
    query_satisfied: bool = Field(
        description="True if there was enough context given to provide an answer "
        "that satisfies the query."
    )


class DefaultRefineProgram(BasePydanticProgram):
    """
    Runs the query on the LLM as normal and always returns the answer with
    query_satisfied=True. In effect, doesn't do any answer filtering.
    """

    def __init__(
        self, prompt: BasePromptTemplate, llm: LLMPredictorType, output_cls: BaseModel
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


class Refine(BaseSynthesizer):
    """Refine a response to a query across text chunks."""

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[BaseModel] = None,
        streaming: bool = False,
        verbose: bool = False,
        structured_answer_filtering: bool = False,
        program_factory: Optional[
            Callable[[BasePromptTemplate], BasePydanticProgram]
        ] = None,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._verbose = verbose
        self._structured_answer_filtering = structured_answer_filtering
        self._output_cls = output_cls

        if self._streaming and self._structured_answer_filtering:
            raise ValueError(
                "Streaming not supported with structured answer filtering."
            )
        if not self._structured_answer_filtering and program_factory is not None:
            raise ValueError(
                "Program factory not supported without structured answer filtering."
            )
        self._program_factory = program_factory or self._default_program_factory

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_qa_template": self._text_qa_template,
            "refine_template": self._refine_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]
        if "refine_template" in prompts:
            self._refine_template = prompts["refine_template"]

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over chunks."""
        response: Optional[RESPONSE_TEXT_TYPE] = None
        t0 = time.time()
        for text_chunk in text_chunks:
            if prev_response is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                response = self._give_response_single(
                    query_str, text_chunk, **response_kwargs
                )
            else:
                # refine response if possible
                response = self._refine_response_single(
                    prev_response, query_str, text_chunk, **response_kwargs
                )
            prev_response = response
        t1 = time.time()
        logger.debug(f"get_resp times[{len(text_chunks)}] cost[{t1-t0}] avg[{(t1-t0)/len(text_chunks)}]")
        if isinstance(response, str):
            if self._output_cls is not None:
                response = self._output_cls.parse_raw(response)
            else:
                response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _default_program_factory(self, prompt: PromptTemplate) -> BasePydanticProgram:
        if self._structured_answer_filtering:
            from llama_index.program.utils import get_program_for_llm

            return get_program_for_llm(
                StructuredRefineResponse,
                prompt,
                self._service_context.llm,
                verbose=self._verbose,
            )
        else:
            return DefaultRefineProgram(
                prompt=prompt,
                llm=self._service_context.llm,
                output_cls=self._output_cls,
            )

    def _give_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

        response: Optional[RESPONSE_TEXT_TYPE] = None
        program = self._program_factory(text_qa_template)
        # TODO: consolidate with loop in get_response_default
        t0 = time.time()
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if response is None and not self._streaming:
                try:
                    structured_response = cast(
                        StructuredRefineResponse,
                        program(
                            context_str=cur_text_chunk,
                            **response_kwargs,
                        ),
                    )
                    query_satisfied = structured_response.query_satisfied
                    if query_satisfied:
                        response = structured_response.answer
                except ValidationError as e:
                    logger.warning(
                        f"Validation error on structured response: {e}", exc_info=True
                    )
            elif response is None and self._streaming:
                response = self._service_context.llm.stream(
                    text_qa_template,
                    context_str=cur_text_chunk,
                    **response_kwargs,
                )
                query_satisfied = True
            else:
                response = self._refine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                    **response_kwargs,
                )
        t1 = time.time()
        logger.debug(f"give_resp_single times[{len(text_chunks)}] cost[{t1-t0}] avg[{(t1-t0)/len(text_chunks)}]")

        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _refine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 5000).replace("\n", "//")
        logger.debug(f"> Refine context: [{len(fmt_text_chunk)}|{fmt_text_chunk}]")
        if self._verbose:
            print(f"> Refine context: [{len(fmt_text_chunk)}|{fmt_text_chunk}]")

        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )

        # compute available chunk size to see if there is any available space
        # determine if the refine template is too big (which can happen if
        # prompt template + query + existing answer is too large)
        avail_chunk_size = (
            self._service_context.prompt_helper._get_available_chunk_size(
                refine_template
            )
        )

        if avail_chunk_size < 0:
            # if the available chunk size is negative, then the refine template
            # is too big and we just return the original response
            return response

        # obtain text chunks to add to the refine template
        text_chunks = self._service_context.prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

        program = self._program_factory(refine_template)
        t0 = time.time()
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if not self._streaming:
                try:
                    structured_response = cast(
                        StructuredRefineResponse,
                        program(
                            context_msg=cur_text_chunk,
                            **response_kwargs,
                        ),
                    )
                    query_satisfied = structured_response.query_satisfied
                    if query_satisfied:
                        response = structured_response.answer
                except ValidationError as e:
                    logger.warning(
                        f"Validation error on structured response: {e}", exc_info=True
                    )
            else:
                # TODO: structured response not supported for streaming
                if isinstance(response, Generator):
                    response = "".join(response)

                refine_template = self._refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )

                response = self._service_context.llm.stream(
                    refine_template,
                    context_msg=cur_text_chunk,
                    **response_kwargs,
                )
        t1 = time.time()
        logger.debug(f"refine_resp_single times[{len(text_chunks)}] cost[{t1-t0}] avg[{(t1-t0)/len(text_chunks)}]")
        return response

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: Optional[RESPONSE_TEXT_TYPE] = None
        for text_chunk in text_chunks:
            if prev_response is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                response = await self._agive_response_single(
                    query_str, text_chunk, **response_kwargs
                )
            else:
                response = await self._arefine_response_single(
                    prev_response, query_str, text_chunk, **response_kwargs
                )
            prev_response = response
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            if self._output_cls is not None:
                response = self._output_cls.parse_raw(response)
            else:
                response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    async def _arefine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")

        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )

        # compute available chunk size to see if there is any available space
        # determine if the refine template is too big (which can happen if
        # prompt template + query + existing answer is too large)
        avail_chunk_size = (
            self._service_context.prompt_helper._get_available_chunk_size(
                refine_template
            )
        )

        if avail_chunk_size < 0:
            # if the available chunk size is negative, then the refine template
            # is too big and we just return the original response
            return response

        # obtain text chunks to add to the refine template
        text_chunks = self._service_context.prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

        program = self._program_factory(refine_template)
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if not self._streaming:
                try:
                    structured_response = await program.acall(
                        context_msg=cur_text_chunk,
                        **response_kwargs,
                    )
                    structured_response = cast(
                        StructuredRefineResponse, structured_response
                    )
                    query_satisfied = structured_response.query_satisfied
                    if query_satisfied:
                        response = structured_response.answer
                except ValidationError as e:
                    logger.warning(
                        f"Validation error on structured response: {e}", exc_info=True
                    )
            else:
                raise ValueError("Streaming not supported for async")

            if query_satisfied:
                refine_template = self._refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )

        return response

    async def _agive_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

        response: Optional[RESPONSE_TEXT_TYPE] = None
        program = self._program_factory(text_qa_template)
        # TODO: consolidate with loop in get_response_default
        for cur_text_chunk in text_chunks:
            if response is None and not self._streaming:
                try:
                    structured_response = await program.acall(
                        context_str=cur_text_chunk,
                        **response_kwargs,
                    )
                    structured_response = cast(
                        StructuredRefineResponse, structured_response
                    )
                    query_satisfied = structured_response.query_satisfied
                    if query_satisfied:
                        response = structured_response.answer
                except ValidationError as e:
                    logger.warning(
                        f"Validation error on structured response: {e}", exc_info=True
                    )
            elif response is None and self._streaming:
                raise ValueError("Streaming not supported for async")
            else:
                response = await self._arefine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                    **response_kwargs,
                )
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response
