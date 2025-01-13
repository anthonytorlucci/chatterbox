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

# def vector_store_from_database(
#     db_dir: Path,
#     collection_name: str,
#     #embeddings = OllamaEmbeddings(model="llama3.2")
# ):
#     embeddings = OllamaEmbeddings(model="llama3.2")
#     try:
#         assert db_dir.exists()
#         db_dir_str = str(db_dir)

#         persistent_client = PersistentClient(path=db_dir_str)

#         return Chroma(
#             client=persistent_client,
#             collection_name=collection_name,
#             embedding_function=embeddings
#         )
#     except AssertionError:
#         print(f"db_dir: {db_dir} does not exist.")
#         return None
#     except Exception as e:
#         raise e


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
        vdbs: List[Tuple[Path,str]],
        k_results: int = 3,
        fetch_k_results: int = 5  # what does this do? how to get the total number of documents in vector store?
    ):
        """
        Initialize the VdbsContextRetriever with vector databases and search parameters.

        This constructor attempts to load each specified Chroma vector database and its
        corresponding collection. Any databases that fail to load are silently skipped,
        allowing the retriever to continue operating with the successfully loaded databases.

        Args:
            vdbs (List[Tuple[Path, str]]): List of tuples where each tuple contains:
                - Path: Directory path where the Chroma database is stored
                - str: Name of the collection within the database
            k_results (int, optional): Number of final results to return from each vector
                store during MMR search. Defaults to 3.
            fetch_k_results (int, optional): Number of initial candidates to retrieve
                before MMR reranking for diversity. Should be greater than k_results.
                Defaults to 5.

        Note:
            - Uses OllamaEmbeddings with "llama3.2" model for vector similarity search
            - Failed database loads are silently handled to maintain graceful degradation
            - The retriever can operate as long as at least one database loads successfully
        """
        self._vector_store_databases = []
        # for db_dir, collection_name in vdbs:
        #     try:
        #         vector_store = vector_store_from_database(
        #             db_dir=db_dir,
        #             collection_name=collection_name
        #         )
        #         self._vector_store_databases.append(vector_store)
        #     except:
        #         continue

        self._k_results = k_results
        self._fetch_k_results = fetch_k_results


    def __call__(self, state: dict):
        """
        Retrieve documents.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, vdbs_context, that contains retrieved documents
        """
        logging.info("---RETRIEVE VDBS CONTEXT---")
        research_prompt = state.get("research_prompt")  # KeyError if research_prompt doesn't exist; something has gone very wrong!
        if not self._vector_store_databases:
            return {
                "calling_agent": self.NAME,
                "vdbs_context": [],
                "messages": state["messages"] + [AIMessage(content="VDBS CONTEXT RETRIEVAL: VECTOR STORE DATABASE INVALID")]
            }

        vdbs_context = []
        for vector_store in self._vector_store_databases:
            retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": self._k_results, "fetch_k": self._fetch_k_results}
            )
            results = retriever.invoke(research_prompt)
            for doc in results:
                vdbs_context.append(doc)
        return {
            "calling_agent": self.NAME,
            "vdbs_context": vdbs_context,
            "requires_vdbs_context": False,
            "messages": state["messages"] + [AIMessage(content="VDBS CONTEXT RETIEVAL SUCCESS")]
        }
