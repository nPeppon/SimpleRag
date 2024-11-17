import os
from typing import List, Tuple
import uuid
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain.globals import set_verbose, set_debug

from simple_rag.common import common, ai_utils
from simple_rag.storage.classes.file_based_byte_store import FileBasedByteStore

import logging
import json


class Summarizer:
    def __init__(self, model_name="gpt-3.5-turbo", max_tokens=4000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,  # Safe chunk size
            chunk_overlap=200,  # Overlap for context preservation
            model_name=model_name
        )
        self.llm = ChatOpenAI(model=model_name, max_retries=3, temperature=0)
        self.prompt_template = ChatPromptTemplate.from_template(
            "Summarize the following document:\n\n{doc}"
        )
        self.parser = StrOutputParser()

    def get_summaries(self, docs: List[Document]) -> List[str]:
        set_debug(True)
        set_verbose(True)
        summaries = []
        for doc in docs:
            logging.debug(f"Document: {doc.metadata.get('source_url')}")
            chunks = ai_utils.get_chunks_from_documents(
                [doc], model_name=self.model_name, chunk_size=4000, chunk_overlap=200)
            # chunks = self.text_splitter.split_documents([doc])

            chain = (
                RunnablePassthrough()
                # Split the document into chunks
                | RunnableLambda(lambda doc: {
                    "doc": self.text_splitter.split_text(doc.page_content)
                })
                | self.prompt_template
                | self.llm
                | self.parser
            )

            # Use batch to process all documents
            chunk_summaries = chain.batch(chunks, {"max_concurrency": 5})
            summaries.append(self._combine_chunk_summaries_single(chunk_summaries))
        set_debug(False)
        set_verbose(False)
        return summaries

    def _combine_chunk_summaries_single(self, chunk_summaries: List[str]) -> str:
        """Combine summaries of chunks for a single document."""
        return " ".join(chunk_summaries)


class MultiRepresentationStore:
    def __init__(self, vector_store_name: str = "multirepresentation_vecstore", byte_store_name: str = "byte_store"):
        # The vectorstore to use to index the child chunks
        vec_store_path = os.path.join(common.get_data_dir(), "multirepresentation_vecstore")
        self.vectorstore = Chroma(collection_name="summaries",
                                  embedding_function=OpenAIEmbeddings(),
                                  persist_directory=vec_store_path)

        # The storage layer for the parent documents
        byte_store_path = os.path.join(common.get_data_dir(), "byte_store")
        self.byte_store = FileBasedByteStore(directory=byte_store_path)
        self.id_key = "doc_id"

        # The retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key=self.id_key,
        )

    def extract_doc_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_url"] = url
            doc.metadata["type"] = "whole_document"
        return docs

    def extract_doc_from_youtube(self, url: str) -> List[Document]:
        video_id = url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine transcript into a single string
        transcript_text = " ".join([entry['text'] for entry in transcript])

        # Create a LangChain document
        doc = Document(page_content=transcript_text, metadata={"source_url": url, "type": "whole_document"})

        return [doc]

    def load_all_documents(self, webpages_urls, youtube_urls) -> List[Document]:
        docs = []
        for url in webpages_urls:
            docs.extend(self.extract_doc_from_url(url))
        for url in youtube_urls:
            docs.extend(self.extract_doc_from_youtube(url))
        return docs

    def get_summaries(self, docs: List[Document]) -> List[str]:
        summaries = Summarizer().get_summaries(docs)
        # summaries = chain.batch(docs, {"max_concurrency": 5})
        return summaries

    def fill_db(self, webpages_urls: List[str], youtube_urls: List[str]):
        existing_urls = set()
        keys = list(self.retriever.byte_store.yield_keys())
        old_docs = self.retriever.byte_store.mget(keys)
        for doc in old_docs:
            if doc:
                try:
                    # Decode the byte data to a string
                    doc_str = doc.decode('utf-8')
                    # Parse the string into a dictionary
                    doc_dict = json.loads(doc_str)

                    existing_urls.add(doc_dict["kwargs"]["metadata"].get("source_url"))
                except Exception as e:
                    logging.error(f"Error getting source_url {doc}: {e}")

        new_docs = []
        for url in webpages_urls:
            if url not in existing_urls:
                new_docs.extend(self.extract_doc_from_url(url))
        for url in youtube_urls:
            if url not in existing_urls:
                new_docs.extend(self.extract_doc_from_youtube(url))

        if not new_docs:
            logging.warning("No new documents to add.")
            return

        new_docs = self.load_all_documents(webpages_urls, youtube_urls)
        doc_ids = [str(uuid.uuid4()) for _ in new_docs]

        for j, doc in enumerate(new_docs):
            doc_chunks = ai_utils.get_chunks_from_documents([doc])
            chunks_id = [str(uuid.uuid4()) for _ in doc_chunks]
            chunks = [
                Document(page_content=chunk.page_content, metadata={
                         **doc.metadata, self.id_key: chunks_id[i], "type": "chunk", "original_doc_id": doc_ids[j]})
                for i, chunk in enumerate(doc_chunks)
            ]
            self.retriever.vectorstore.add_documents(chunks)
            # Persist chunks in the byte store
            self.retriever.docstore.mset(list(zip(chunks_id, chunks)))

        summaries = self.get_summaries(new_docs)
        # Docs linked to summaries
        summary_docs = [
            Document(page_content=s, metadata={
                     self.id_key: doc_ids[i], "type": "summary", "source_url": new_docs[i].metadata.get("source_url", "")})
            for i, s in enumerate(summaries)
        ]

        # Add
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, new_docs)))

        # Persist the vectorstore and byte store
        self.vectorstore.persist()
        self.byte_store.save()

    def retrieve_original_text(self, query: str, score_threshold: float = .65, num_results: int = 5) -> List[Tuple[str, str, Document]]:
        types = ["whole_document", "summary"]
        type_filter = {"type": {"$in": types}}
        results = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold, "filter": type_filter}).get_relevant_documents(
            query, n_results=num_results)
        if results:
            logging.debug(f"Retrieved {len(results)} results")
            logging.debug([r.metadata for r in results])
            return [(r.metadata.get("source_url", ""), r.metadata.get("doc_id", ""), r) for r in results]
        logging.warning("No results found.")
        return []

    def retrieve_specific_chunks(self, query: str, source_url: str = "", original_doc_id: str = "", num_results: int = 5, score_threshold: float = .65) -> List[Tuple[str, Document]]:
        filter = {"type": "chunk"}
        if original_doc_id:
            original_doc_id_filter = {"original_doc_id": original_doc_id}
            filter = {"$and": [filter, original_doc_id_filter]}
        if source_url:
            source_url_filter = {"source_url": source_url}
            filter = {"$and": [filter, source_url_filter]}
        results = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold, "filter": filter}).get_relevant_documents(
            query, n_results=num_results)
        if results:
            logging.debug(f"Retrieved {len(results)} results")
            logging.debug([r.metadata for r in results])
            return [(r.metadata.get("source_url", ""), r) for r in results]
        logging.warning("No results found.")
        return []

    def retrieve_chunks_from_scratch(self, query: str, score_threshold: float = .65, num_results: int = 5) -> List[str]:
        """
        Retrieve chunks from scratch by first retrieving the original text and then the chunks.

        Args:
            query (str): The query to search for.
            score_threshold (float): The similarity score threshold to use.
            num_results (int): The number of results to retrieve.
        """
        results = self.retrieve_original_text(query, score_threshold, num_results=num_results)
        if not results:
            logging.warning("No results found.")
            return []
        chunks_list = []
        for result in results:
            logging.debug(f"Retrieving chunks for {result[0]}")
            chunks = self.retrieve_specific_chunks(
                query, source_url=result[0], num_results=num_results, score_threshold=score_threshold)
            if chunks:
                chunks_list.extend(chunk[1].page_content for chunk in chunks)
        return chunks_list

    def list_all_metadata(self):
        # Retrieve all documents from the vector store
        all_documents = self.vectorstore.get()

        # Extract and print metadata for each document
        logging.debug(f"All Metadata: {all_documents.get('metadatas')}")

    def get_original_text_retriever(self, score_threshold: float = .65) -> MultiVectorRetriever:
        types = ["whole_document", "summary"]
        type_filter = {"type": {"$in": types}}
        return self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold, "filter": type_filter})

    def get_chunk_retriever(self, score_threshold: float = .65, original_doc_id: str = "", source_url: str = "") -> MultiVectorRetriever:
        filter = {"type": "chunk"}
        if original_doc_id:
            original_doc_id_filter = {"original_doc_id": original_doc_id}
            filter = {"$and": [filter, original_doc_id_filter]}
        if source_url:
            source_url_filter = {"source_url": source_url}
            filter = {"$and": [filter, source_url_filter]}
        return self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold, "filter": {"type": "chunk"}})

    def retrieve_and_format(query):
        storage = MultiRepresentationStore()
        chunks = storage.retrieve_chunks_from_scratch(query)
        return "\n".join(chunks)

if __name__ == "__main__":
    webpages_urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
                     "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"]
    youtube_urls = ["https://www.youtube.com/watch?v=sVcwVQRHIc8"]
    storage = MultiRepresentationStore()
    storage.fill_db(webpages_urls, youtube_urls)

    results = storage.retrieve_original_text("What is a multirepresentation indexing?")
    print(results[0][0])
    print(results[0][2].page_content)
    storage.list_all_metadata()
    specific_results = storage.retrieve_specific_chunks(
        "What is a multirepresentation indexing?", source_url=results[0][0])
    print(specific_results[0][1].page_content)
