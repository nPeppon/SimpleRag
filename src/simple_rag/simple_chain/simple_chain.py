from typing import List
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging
from simple_rag.storage.multi_representation_storage import MultiRepresentationStore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_list_of_strings(strings: List[str]):
    return "\n".join(strings)

def retrieve_and_format(query):
    storage = MultiRepresentationStore()
    chunks = storage.retrieve_chunks_from_scratch(query)
    return format_list_of_strings(chunks)

def call_rag_simple_chain(question: str) -> str:
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # Chain
    rag_chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return result


if __name__ == "__main__":
    question = "What is a multirepresentation indexing?"
    response = call_rag_simple_chain(question)
    print(response)