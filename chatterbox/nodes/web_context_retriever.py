"""
web_context_retriever

Retrieves textual context from a vector store created from a list of urls.
"""

# standard lib
from uuid import uuid4
from typing import List

# third party
import chromadb

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_community.document_loaders import WebBaseLoader
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

class WebContextRetriever(ResearcherInterface):
    """
    A LangGraph agent that retrieves relevant contextual information from web pages using vector similarity search.

    This agent creates a vector store from a list of provided URLs by:
    1. Loading web page content
    2. Splitting content into chunks
    3. Creating embeddings using Ollama
    4. Storing embeddings in a Chroma vector database

    When called, it retrieves relevant context based on main ideas present in the graph state.

    Parameters
    ----------
    urls : List[str]
        List of URLs to load and process for context retrieval
    chunk_size : int, optional
        Size of text chunks when splitting documents (default=400)
    chunk_overlap : int, optional
        Number of characters to overlap between chunks (default=100)
    k_results : int, optional
        Number of similar documents to retrieve per query (default=3)

    Attributes
    ----------
    NAME : str
        Identifier for the agent in the graph
    _embeddings : OllamaEmbeddings
        Embedding model instance
    _vector_store : Chroma
        Vector database instance
    _k_results : int
        Number of results to retrieve per query

    State Contract
    -------------
    Input state keys:
        - main_ideas: List of ideas to search for relevant context

    Output state keys:
        - calling_agent: Name of this agent
        - web_context: Retrieved document chunks
        - requires_web_context: Set to False after retrieval
    """
    NAME = "web_context_retriever"
    def __init__(self,
        urls: List[str],
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        k_results: int = 3
    ):
        """
        Initialize the WebContextRetriever agent.

        Parameters
        ----------
        urls : List[str]
            List of URLs to load and process for context retrieval
        chunk_size : int, optional
            Size of text chunks when splitting documents (default=400)
        chunk_overlap : int, optional
            Number of characters to overlap between chunks (default=100)
        k_results : int, optional
            Number of similar documents to retrieve per query (default=3)

        Notes
        -----
        The agent uses Ollama embeddings with the "llama3.2" model and stores vectors in a Chroma database.
        """
        self._urls = urls
        self._k_results = k_results
        # TODO: add to parameters
        self._embeddings = OllamaEmbeddings(model="llama3.2")

        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self._vector_store = Chroma(
            collection_name="web_context_retriever_collection",
            embedding_function=self._embeddings,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if self._urls:
            documents = []
            for url in self._urls:
                loader = WebBaseLoader(
                    web_path=url,
                    autoset_encoding=False,
                    encoding="utf-8",
                    continue_on_failure=True
                )
                docs, _ = process_items_safely(loader.lazy_load())
                documents.extend(docs)


            all_splits = self._text_splitter.split_documents(documents)

            uuids = [str(uuid4()) for _ in range(len(all_splits))]
            if self._k_results > len(all_splits):
                self._k_results = len(all_splits)
            self._vector_store.add_documents(documents=all_splits, ids=uuids)


    def __call__(self, state: dict):
        """
        Retrieve documents from provided urls.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, pdf_context, that contains retrieved documents
        """
        #---print("---RETRIEVE WEB CONTEXT---")
        #---research_prompt = state.get("research_prompt", "")
        web_context = state.get("web_context", [])
        main_ideas = state.get("main_ideas", [])

        try:
            if not self._urls:
                raise ValueError()  # supervisor routed to retriever, but no urls were given
            for idea in main_ideas:
                ### search by vector
                results = self._vector_store.similarity_search_by_vector(
                    embedding=self._embeddings.embed_query(idea),
                    k=self._k_results
                )
                for doc in results:
                    web_context.append(doc)
            return {
                "calling_agent": self.NAME,
                "web_context": web_context,
                "messages": state["messages"] + [AIMessage(content="WEB CONTEXT RETRIEVAL: SUCCESS")]
            }
        except:
            return {
                "calling_agent": self.NAME,
                "web_context": web_context,
                "messages": state["messages"] + [AIMessage(content="WEB CONTEXT RETRIEVAL: FAILED")]
            }
