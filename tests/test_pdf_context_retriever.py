"""

"""
# standard lib
from pathlib import Path
# third party
import pytest
# langchain
# from langchain_core.messages.ai import AIMessage
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# langgraph
# local
from chatterbox.nodes.pdf_context_retriever import PdfContextRetriever

PDF_FILE_PATH = Path(__file__).parent.joinpath("3490443.pdf")

@pytest.fixture
def pdf_retriever():
    """Fixture to create a basic ArxivContextRetriever instance."""
    return PdfContextRetriever(
        pdf_paths=[PDF_FILE_PATH],
        chunk_size=200,
        chunk_overlap=50,
        k_results=2
    )

class TestArxivContextRetriever:

    def test_initialization(self):
        """Test if the PdfContextRetriever initializes with correct parameters."""
        retriever = PdfContextRetriever(
            pdf_paths=[PDF_FILE_PATH],
            chunk_size=300,
            chunk_overlap=75,
            k_results=2
        )

        assert retriever._vector_store_is_valid == True
        assert retriever._k_results == 2

    def test_init_invalid_pdf_paths(self):
        # Arrange
        pdf_paths = [Path("path/to/non_existent_pdf.pdf")]

        # Act and Assert
        with pytest.raises(ValueError):
            PdfContextRetriever(pdf_paths, chunk_size=400, chunk_overlap=100, k_results=3)

    def test_empty_initialization(self):
        """Test if the PdfContextRetreiver initializes with empty pdf paths."""
        retriever = PdfContextRetriever(
            pdf_paths=[],
            chunk_size=300,
            chunk_overlap=75,
            k_results=2
        )

        assert retriever._vector_store_is_valid == False
