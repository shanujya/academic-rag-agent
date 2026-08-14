"""Self-RAG LangGraph workflow assembly."""

from langgraph.graph import END, START, StateGraph

from config import MAX_GENERATION_RETRIES, MAX_RETRIEVE_CYCLES
from src.agent import nodes
from src.agent.state import AgentState


def route_after_grade_documents(state: AgentState) -> str:
    relevant = state.get("relevant_documents") or []
    if relevant:
        return "generate"
    return "web_search"


def route_after_grade_generation(state: AgentState) -> str:
    result = state.get("grade_generation_result") or {}
    grounded = result.get("grounded", "no").lower() == "yes"
    retries = state.get("generation_retries", 0)

    if grounded:
        return "grade_answer"
    if retries >= MAX_GENERATION_RETRIES:
        return "grade_answer"
    return "generate"


def route_after_grade_answer(state: AgentState) -> str:
    result = state.get("grade_answer_result") or {}
    addresses = result.get("addresses_question", "no").lower() == "yes"
    cycles = state.get("retrieve_cycles", 0)

    if addresses:
        return "finish"
    if cycles >= MAX_RETRIEVE_CYCLES:
        return "finish"
    return "retrieve"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("generate", nodes.generate)
    graph.add_node("grade_generation", nodes.grade_generation)
    graph.add_node("grade_answer", nodes.grade_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {"generate": "generate", "web_search": "web_search"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges(
        "grade_generation",
        route_after_grade_generation,
        {"generate": "generate", "grade_answer": "grade_answer"},
    )
    graph.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {"finish": END, "retrieve": "retrieve"},
    )

    return graph.compile()


def run_agent(question: str) -> dict:
    app = build_graph()
    initial: AgentState = {
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "web_context": "",
        "generation": "",
        "steps_log": [],
        "generation_retries": 0,
        "retrieve_cycles": 0,
        "grade_documents_result": {},
        "grade_generation_result": {},
        "grade_answer_result": {},
    }
    return app.invoke(initial)
