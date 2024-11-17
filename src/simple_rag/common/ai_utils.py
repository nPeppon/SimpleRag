from typing import List
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Compute the number of tokens in the given text for the specified model.
       Possible models inclue "gpt-3.5-turbo" and "gpt-4-turbo".

         Args:
              text (str): The text to count tokens for.
              model_name (str): The model to use for tokenization. Default is "gpt-3.5-turbo".
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def get_chunks_from_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 200, model_name: str = "gpt-3.5-turbo") -> List[str]:
    """Split the given text into chunks of the specified size and overlap.

       Args:
            text (str): The text to split into chunks.
            chunk_size (int): The size of each chunk. Default is 4000.
            chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.
            model_name (str): The model to use for tokenization. default is "gpt-3.5-turbo".

       Returns:
            List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name
    )
    return text_splitter.split_text(text)


def get_chunks_from_documents(docs: List[Document], chunk_size: int = 4000, chunk_overlap: int = 200, model_name: str = "gpt-3.5-turbo") -> List[str]:
    """Split the given list of documents into chunks of the specified size and overlap.

       Args:
            docs (List[str]): The list of documents to split into chunks.
            chunk_size (int): The size of each chunk. Default is 4000.
            chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.
            model_name (str): The model to use for tokenization. default is "gpt-3.5-turbo".

       Returns:
            List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name
    )
    return text_splitter.split_documents(docs)
