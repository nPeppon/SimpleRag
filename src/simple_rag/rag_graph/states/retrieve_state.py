from langchain.schema import Document
from simple_rag.rag_graph.states.graph_state import GraphState
from simple_rag.storage.multi_representation_storage import MultiRepresentationStore

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    storage = MultiRepresentationStore()
    # Retrieval
    documents = storage.retrieve_chunks_from_scratch(question)
    documents = [Document(page_content=doc) for doc in documents]
    # documents = retriever.invoke(question)
    return {"documents": documents, "question": question}