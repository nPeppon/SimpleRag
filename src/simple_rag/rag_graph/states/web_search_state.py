from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from simple_rag.rag_graph.states.graph_state import GraphState

def web_search(state: GraphState) -> GraphState:
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = [Document(page_content=web_results)]

    return {"documents": web_results, "question": question}
