"""LangGraph agent state definitions."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    relevant_documents: list[Document]
    web_context: str
    generation: str
    steps_log: Annotated[list[str], operator.add]
    generation_retries: int
    retrieve_cycles: int
    grade_documents_result: dict
    grade_generation_result: dict
    grade_answer_result: dict
