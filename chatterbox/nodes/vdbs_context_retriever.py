"""
vdbs_context_retriever

Retrieves textual context from a pre-built chroma database as a vector store.
"""

# standard lib
from typing import List, Tuple
from pathlib import Path
import logging

# third party
# from pydantic import BaseModel, Field
from chromadb import PersistentClient

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# langgraph

# local
from chatterbox.researcher_interface import (
    ResearcherInterface
)


### subclass ResearcherInterface

class VdbsContextRetriever(ResearcherInterface):
    """
    A LangGraph agent that retrieves relevant documents from pre-built vector stores.

    This agent performs semantic search and retrieval of documents from multiple
    Chroma vector store databases that have been previously built and saved to disk.
    It uses Maximum Marginal Relevance (MMR) search to retrieve diverse, relevant context
    based on the research prompt in the current state.

    Attributes:
        NAME (str): Identifier for the agent in the multi-agent system
        _vector_store_databases (list): List of loaded Chroma vector store instances
        _k_results (int): Number of results to return from each vector store
        _fetch_k_results (int): Number of initial candidates to fetch before MMR reranking

    Args:
        vdbs (List[Tuple[Path, str]]): List of tuples containing (database_directory, collection_name)
        k_results (int, optional): Number of results to return from each vector store. Defaults to 3.
        fetch_k_results (int, optional): Number of initial candidates to fetch before MMR reranking. Defaults to 5.

    Example:
        ```python
        retriever = VdbsContextRetriever(
            vdbs=[(Path("./db"), "research_papers")],
            k_results=3
        )
        new_state = retriever(current_state)
        ```

    Returns:
        When called, returns a dict with:
            - calling_agent: Name of this agent
            - vdbs_context: List of retrieved documents
            - requires_vdbs_context: Flag set to False after retrieval
            - messages: Updated message history with retrieval status
    """
    NAME = "vdbs_context_retriever"
    def __init__(self,
        db_dir: Path,
        collection_names: List[str],
        k_results: int = 3,
    ):
        """
        Initialize the VdbsContextRetriever with vector databases and search parameters.

        Args:
            vdbs (List[Tuple[Path, str]]): List of tuples where each tuple contains:
                - Path: Directory path where the Chroma database is stored
                - str: Name of the collection within the database
            k_results (int, optional): Number of final results to return from each vector
                store during MMR search. Defaults to 3.


        Note:
            - Uses OllamaEmbeddings with "llama3.2" model for vector similarity search
            - Failed database loads are silently handled to maintain graceful degradation
            - The retriever can operate as long as at least one database loads successfully
        """
        self._k_results = k_results
        self._collection_names = collection_names

        # TODO: add to parameters
        self._embeddings = OllamaEmbeddings(model="llama3.2")
        self._db_dir = db_dir



    def __call__(self, state: dict):
        """
        Retrieve documents.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, vdbs_context, that contains retrieved documents
        """
        logging.info("---RETRIEVE VDBS CONTEXT---")
        main_ideas = state.get("main_ideas", [])
        vdbs_context = state.get("vdbs_context", [])
        if any([len(self._collection_names)==0, not self._db_dir.exists()]):
            return {
                "calling_agent": self.NAME,
                "vdbs_context": [],
                "requires_vdbs_context": False,
                "messages": state["messages"] + [AIMessage(content="VDBS CONTEXT RETRIEVAL: VECTOR STORE DATABASE INVALID")]
            }

        persistent_client = PersistentClient(path=str(self._db_dir))
        for collection_name in self._collection_names:
            try:
                vector_store = Chroma(
                    client = persistent_client,
                    collection_name=collection_name,
                    embedding_function=self._embeddings
                )
                for idea in main_ideas:
                    ### search by vector
                    results = vector_store.similarity_search_by_vector(
                        embedding=self._embeddings.embed_query(idea),
                        k=self._k_results
                    )
                    for doc in results:
                        vdbs_context.append(doc)
            except:
                continue
        return {
            "calling_agent": self.NAME,
            "vdbs_context": vdbs_context,
            "requires_vdbs_context": False,
            "messages": state["messages"] + [AIMessage(content="VDBS CONTEXT RETIEVAL SUCCESS")]
        }
