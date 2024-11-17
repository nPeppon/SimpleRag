from typing import Dict, List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    
    def get_value_for_logger(self, max_length: int = 1000) -> Dict:
        new_value = self.copy()
        if 'documents' in new_value:
            if len(new_value['documents']) > max_length:
                new_value['documents'] = new_value['documents'][:max_length] + ['... (truncated)']
        return new_value

    def __str__(self) -> str:
        return self. get_value_for_logger().__str__()
