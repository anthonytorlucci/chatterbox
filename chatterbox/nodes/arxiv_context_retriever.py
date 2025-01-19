"""
arxiv_context_retriever

Retrieves textual context from a vector store created from a list of arxiv references..
"""

# standard lib
from uuid import uuid4
import logging

# third party
import chromadb

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_community.document_loaders import ArxivLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# langgraph

# local
from chatterbox.researcher_interface import (
    ResearcherInterface
)
from chatterbox.utils import process_items_safely


### subclass ResearcherInterface

class ArxivContextRetriever(ResearcherInterface):
    """
    A LangGraph agent that retrieves relevant academic papers from arXiv based on provided research ideas.

    This class loads, processes, and retrieves contextual information from arXiv papers using vector similarity search.
    It creates embeddings of document chunks and stores them in a Chroma vector store for efficient retrieval.

    Parameters:
        max_docs_to_load (int): Maximum number of documents to load per research idea. Defaults to 10.
        chunk_size (int): Size of text chunks for document splitting. Defaults to 400.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        k_results (int): Number of most relevant results to return per query. Defaults to 3.

    Attributes:
        _embeddings: Ollama embeddings model for converting text to vectors
        _vector_store: Chroma vector store for storing and retrieving document embeddings
        _text_splitter: Recursive character text splitter for breaking documents into chunks

    The agent processes the state dictionary containing "main_ideas" and adds retrieved arXiv context
    under the "arxiv_context" key. The retrieved documents can then be evaluated by the ResearchContextGrader
    agent for relevance and quality assessment.

    Example:
        retriever = ArxivContextRetriever(max_docs_to_load=5, k_results=3)
        state = {"main_ideas": ["quantum computing algorithms"]}
        new_state = retriever(state)
    """
    NAME = "arxiv_context_retriever"
    def __init__(self,
        max_docs_to_load: int = 10,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        k_results: int = 3
    ):
        """
        Initialize the ArxivContextRetriever with specified parameters.

        Args:
            max_docs_to_load (int, optional): Maximum number of arXiv documents to load per research idea.
                Defaults to 10.
            chunk_size (int, optional): Size of text chunks when splitting documents for embedding.
                Larger chunks preserve more context but use more memory. Defaults to 400.
            chunk_overlap (int, optional): Number of characters that overlap between consecutive chunks
                to maintain context across chunk boundaries. Defaults to 100.
            k_results (int, optional): Number of most relevant document chunks to retrieve per query.
                Defaults to 3.

        Initializes:
            - Ollama embeddings model for text vectorization
            - Chroma vector store for document storage and retrieval
            - RecursiveCharacterTextSplitter for document chunking
        """

        self._k_results = k_results
        self._max_docs_to_load = max_docs_to_load
        # TODO: add to parameters
        self._embeddings = OllamaEmbeddings(model="llama3.2")

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self._vector_store = Chroma(
            collection_name="arxiv_context_retriever_collection",
            embedding_function=self._embeddings,
            # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def __call__(self, state: dict):
        """
        Retrieve documents from arxiv.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, pdf_context, that contains retrieved documents
        """
        logging.info("---RETRIEVE ARXIV CONTEXT---")
        arxiv_context = state.get("arxiv_context", [])

        if main_ideas := state.get("main_ideas", ""):
            for idea in main_ideas:
                loader = ArxivLoader(
                    query=idea,
                    load_max_docs=self._max_docs_to_load,
                    doc_content_chars_max=1000,
                    load_all_available_meta=False,
                    # ...
                )
                documents, _ = process_items_safely(loader.lazy_load())
                all_splits = self._text_splitter.split_documents(documents)

                uuids = [str(uuid4()) for _ in range(len(all_splits))]
                if self._k_results > len(all_splits):
                    self._k_results = len(all_splits)
                self._vector_store.add_documents(documents=all_splits, ids=uuids)

                ### search by vector
                results = self._vector_store.similarity_search_by_vector(
                    embedding=self._embeddings.embed_query(idea), k=self._k_results
                )
                for doc in results:
                    arxiv_context.append(doc)
            return {"calling_agent": self.NAME, "arxiv_context": arxiv_context, "requires_arxiv_context": False, "messages": state["messages"] + [AIMessage(content="ARXIV CONTEXT RETRIEVED")]}
        else:
            return {"calling_agent": self.NAME, "arxiv_context": [], "messages": state["messages"] + [AIMessage(content="NO ARXIV CONTEXT RETRIEVED")]}
