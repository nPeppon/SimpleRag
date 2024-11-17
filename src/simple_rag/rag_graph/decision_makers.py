from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import ClassVar

class RouterAI(ChatOpenAI):
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["vectorstore", "web_search"] = Field(
            ...,
            description="Given a user question choose to route it to web search or a vectorstore.",
        )
        
    system_prompt: ClassVar[str] = """You are an expert at routing a user question to a vectorstore or web search.
                        The vectorstore contains documents related to agents, prompt engineering, 
                        adversarial attacks and general knowledge on RAG systems.
                        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
                        
    route_prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def structured_llm(self):
        return self.with_structured_output(self.RouteQuery)
    
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

class HallucinationGraderAI(ChatOpenAI):
    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
    # Prompt
    system_prompt: ClassVar[str] = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def structured_llm(self):
        return self.with_structured_output(self.GradeHallucinations)


### Answer Grader
class AnswerGraderAI(ChatOpenAI):

    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    # Prompt
    system: ClassVar[str] = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def structured_llm(self):
        return self.with_structured_output(self.GradeAnswer)



question_router_ai = RouterAI(model="gpt-3.5-turbo-0125", temperature=0)
question_router = question_router_ai.route_prompt | question_router_ai.structured_llm

doc_grader_ai = DocGraderAI(model="gpt-3.5-turbo-0125", temperature=0)
doc_grader = doc_grader_ai.grade_prompt | doc_grader_ai.structured_llm

hallucination_grader_ai = HallucinationGraderAI(model="gpt-3.5-turbo-0125", temperature=0)
hallucination_grader = hallucination_grader_ai.hallucination_prompt | hallucination_grader_ai.structured_llm

answer_grader_ai = AnswerGraderAI(model="gpt-3.5-turbo-0125", temperature=0)
answer_grader = answer_grader_ai.answer_prompt | answer_grader_ai.structured_llm