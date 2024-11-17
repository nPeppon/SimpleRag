from typing import ClassVar
from langchain_openai import ChatOpenAI
from simple_rag.rag_graph.states.graph_state import GraphState
from langchain_core.prompts import ChatPromptTemplate


class QuestionRewriterAI(ChatOpenAI):
    """
    Question rewriter AI that rewrites a question to be optimized for vectorstore retrieval or for web search
    """
    # Prompt
    system_prompt: ClassVar[str] = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def transform_query(state: GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    question_rewriter = QuestionRewriterAI(model="gpt-3.5-turbo-0125", temperature=0)
    question_rewriter = question_rewriter.re_write_prompt | question_rewriter
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question.content}