from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import ClassVar, List

from simple_rag.rag_graph.states.graph_state import GraphState



class DocGraderAI(ChatOpenAI):
    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
        
    # Prompt
    system_prompt: ClassVar[str] = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def structured_llm(self):
        return self.with_structured_output(self.GradeDocuments)


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    doc_grader_llm = DocGraderAI(model="gpt-3.5-turbo-0125", temperature=0)
    retrieval_grader = doc_grader_llm.grade_prompt | doc_grader_llm.structured_llm
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}