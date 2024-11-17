from typing import Dict
from langgraph.graph import END, StateGraph, START

from simple_rag.rag_graph.states.graph_state import GraphState
from simple_rag.rag_graph.states.web_search_state import web_search
from simple_rag.rag_graph.states.retrieve_state import retrieve
from simple_rag.rag_graph.states.doc_grader_state import grade_documents
from simple_rag.rag_graph.states.generate_state import generate
from simple_rag.rag_graph.states.query_transform_state import transform_query

from simple_rag.rag_graph.edges.edges import route_question, decide_to_generate, grade_generation_v_documents_and_question
import logging


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

def get_value_for_logger(value: Dict, max_length: int = 1000) -> Dict:
    new_value = value.copy()
    if 'documents' in value:
        if len(value['documents']) > max_length:
            new_value['documents'] = value['documents'][:max_length] + ['... (truncated)']
    return new_value

if __name__ == "__main__":
    app = workflow.compile()
    questions = [
        "What player at the Bears expected to draft first in the 2024 NFL draft?",
        "What is the capital of France?",
        "What is a multirepresentation indexing?",
    ]
    # Run
    inputs = {
        "question": questions[2]
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            logging.info(f"Node '{key}':")
            # Optional: print full state at each node
            logging.debug(value)
        logging.info("\n---\n")

    # Final generation
    logging.info(value["generation"])