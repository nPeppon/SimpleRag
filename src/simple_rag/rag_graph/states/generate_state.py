from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from simple_rag.rag_graph.states.graph_state import GraphState

def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    documents = "\n\n".join(doc.page_content for doc in documents)
    # RAG generation    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}