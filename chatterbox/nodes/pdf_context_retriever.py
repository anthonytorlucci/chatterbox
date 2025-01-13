"""
pdf_context_retriever

Retrieves textual context from a vector store created from a list of pdf files.
"""

# standard lib
from uuid import uuid4
from pathlib import Path
from typing import List
import logging

# third party
# from pydantic import BaseModel, Field

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# langgraph

# local
from chatterbox.researcher_interface import (
    ResearcherInterface
)
from chatterbox.utils import (
    process_items_safely
)


### subclass ResearcherInterface

class PdfContextRetriever(ResearcherInterface):
    """
    A document retrieval agent that processes PDF files and retrieves relevant context based on main ideas.

    This class is part of a LangGraph multi-agent system that handles PDF document processing
    and context retrieval. It loads PDF documents, splits them into chunks, creates embeddings,
    and stores them in a vector store for semantic similarity search.

    Parameters:
        pdf_paths (List[Path]): List of paths to PDF files to process
        pages (Optional[List[Tuple[int,int]]], optional): List of page ranges (start, end) for each PDF.
            If None, processes all pages. Defaults to None.
        chunk_size (int, optional): Size of text chunks for document splitting. Defaults to 400.
        chunk_overlap (int, optional): Overlap size between chunks. Defaults to 100.
        k_results (int, optional): Number of similar documents to retrieve per query. Defaults to 3.

    Attributes:
        NAME (str): Identifier for the agent, set to "pdf_context_retriever"
        _vector_store_is_valid (bool): Indicates if the vector store was successfully created
        _embeddings: Ollama embeddings model instance
        _vector_store: Chroma vector store instance
        _k_results (int): Number of results to retrieve per query

    The class processes PDFs on initialization and provides a __call__ method that retrieves
    relevant context based on main ideas present in the graph state. Retrieved context is later
    used by the ResearchContextGrader for evaluation and analysis.
    """
    NAME = "pdf_context_retriever"
    def __init__(self,
        pdf_paths: List[Path],
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        k_results: int = 3
    ):
        """
        Initialize the PDF context retriever with document processing and vector store setup.

        Args:
            pdf_paths (List[Path]): List of paths to PDF files to process and index.
            chunk_size (int, optional): The size of text chunks when splitting documents.
                Larger chunks provide more context but may reduce retrieval precision. Defaults to 400.
            chunk_overlap (int, optional): The number of characters to overlap between chunks.
                Helps maintain context across chunk boundaries. Defaults to 100.
            k_results (int, optional): The number of similar documents to retrieve per query.
                Cannot exceed the total number of chunks. Defaults to 3.

        Note:
            - The initializer processes all PDFs, creates embeddings, and stores them in a Chroma vector database
            - If no documents are successfully loaded, the vector store will be marked as invalid
            - Uses Ollama embeddings with the llama3.2 model for document vectorization
            - Generates unique UUIDs for each document chunk for vector store indexing
        """
        # TODO: assert each item in pdf_paths is a Path object or can be converted to a Path object
        self._vector_store_is_valid = False

        documents = []
        for i,pdf_path in enumerate(pdf_paths):
            loader = PyMuPDFLoader(file_path=str(pdf_path), extract_images=False)
            docs, _ = process_items_safely(loader.lazy_load())
            documents.extend(docs)

        # if documents is empty, there is nothing to store and the node cannot be called
        if documents:
            # TODO: add to parameters
            self._embeddings = OllamaEmbeddings(model="llama3.2")

            self._vector_store = Chroma(
                collection_name="pdf_context_retriever_collection",
                embedding_function=self._embeddings,
                # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_splits = text_splitter.split_documents(documents)

            uuids = [str(uuid4()) for _ in range(len(all_splits))]
            if k_results > len(all_splits):
                self._k_results = len(all_splits)
            else:
                self._k_results = k_results

            self._vector_store.add_documents(documents=all_splits, ids=uuids)
            self._vector_store_is_valid = True


    def __call__(self, state: dict):
        """
        Retrieve documents.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, pdf_context, that contains retrieved documents
        """
        logging.info("---RETRIEVE PDF CONTEXT---")
        pdf_context = state.get("pdf_context", [])
        if not self._vector_store_is_valid:
            return {"calling_agent": self.NAME, "pdf_context": []}

        if main_ideas := state.get("main_ideas", ""):
            #---pdf_context = []
            for idea in main_ideas:
                ### search by vector
                results = self._vector_store.similarity_search_by_vector(
                    embedding=self._embeddings.embed_query(idea), k=self._k_results
                )
                for doc in results:
                    # for sentence in split_into_sentences(doc.page_content):  # is this necessary? this might be harmful and lacking proper context from surrounding sentences.
                    #     pdf_context.append(sentence)
                    doc.page_content = str(doc.page_content).replace("\n", " ")
                    pdf_context.append(doc)
            return {
                "calling_agent": self.NAME,
                "pdf_context": pdf_context,
                "requires_pdf_context": False,
                "messages": state["messages"] + [AIMessage(content="PDF CONTEXT RETRIEVED")]
            }
        else:
            return {
                "calling_agent": self.NAME,
                "pdf_context": [],
                "requires_pdf_context": False,
                "messages": state["messages"] + [AIMessage(content="PDF CONTEXT RETRIEVED")]
            }
