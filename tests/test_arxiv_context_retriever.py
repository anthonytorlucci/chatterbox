"""
The ArxivContextRetriever class is designed to retrieve academic papers from
arXiv based on provided research ideas. It uses several external libraries
like LangChain for embeddings, Chroma for vector storage, and ArxivLoader for
fetching documents. The class processes these documents into chunks, stores
them in a vector store, and retrieves relevant contexts based on user queries.

These tests ...
"""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages.ai import AIMessage
from langchain_core.documents import Document
from chatterbox.nodes.arxiv_context_retriever import ArxivContextRetriever
from chatterbox.utils import process_items_safely

@pytest.fixture
def arxiv_retriever():
    """Fixture to create a basic ArxivContextRetriever instance."""
    return ArxivContextRetriever(
        max_docs_to_load=3,
        chunk_size=200,
        chunk_overlap=50,
        k_results=2
    )

@pytest.fixture
def mock_documents():
    """Fixture to create mock document objects."""
    return [
        Document(page_content="Test content 1", metadata={"title": "Test Paper 1"}),
        Document(page_content="Test content 2", metadata={"title": "Test Paper 2"}),
    ]

class TestArxivContextRetriever:

    def test_initialization(self):
        """Test if the ArxivContextRetriever initializes with correct parameters."""
        retriever = ArxivContextRetriever(
            max_docs_to_load=1,
            chunk_size=300,
            chunk_overlap=75,
            k_results=4
        )

        assert retriever._max_docs_to_load == 1
        assert retriever._k_results == 4
        assert retriever._text_splitter._chunk_size == 300
        assert retriever._text_splitter._chunk_overlap == 75

    @patch('chatterbox.nodes.arxiv_context_retriever.ArxivLoader')
    @patch('chatterbox.nodes.arxiv_context_retriever.process_items_safely')
    def test_call_with_main_ideas(self, mock_process_items, mock_arxiv_loader, arxiv_retriever, mock_documents):
        """Test the __call__ method with valid main ideas."""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_arxiv_loader.return_value = mock_loader_instance
        mock_process_items.return_value = (mock_documents, None)

        # Create test state
        test_state = {
            "main_ideas": ["quantum computing"],
            "messages": []
        }

        # Execute
        result = arxiv_retriever(test_state)

        # Assertions
        assert "arxiv_context" in result
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["calling_agent"] == "arxiv_context_retriever"
        assert not result["requires_arxiv_context"]

    def test_call_without_main_ideas(self, arxiv_retriever):
        """Test the __call__ method when no main ideas are provided."""
        test_state = {
            "messages": []
        }

        result = arxiv_retriever(test_state)

        assert result["arxiv_context"] == []
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "NO ARXIV CONTEXT RETRIEVED"

    # def test_vector_store_integration(self, arxiv_retriever, mock_documents):
    #     """Test the integration with the vector store."""
    #     # Note: This is more of an integration test and might need to be adjusted
    #     # based on your testing environment
    #     test_state = {
    #         "main_ideas": ["machine learning"],
    #         "messages": []
    #     }

    #     with patch('chatterbox.nodes.arxiv_context_retriever.ArxivLoader') as mock_loader:
    #         mock_loader_instance = Mock()
    #         mock_loader.return_value = mock_loader_instance
    #         mock_loader_instance.lazy_load.return_value = mock_documents

    #         result = arxiv_retriever(test_state)

    #         assert len(result["arxiv_context"]) <= arxiv_retriever._k_results

    def test_error_handling(self, arxiv_retriever):
        """Test error handling in the retriever."""
        test_state = {
            "main_ideas": None,  # Invalid input
            "messages": []
        }

        result = arxiv_retriever(test_state)

        assert result["arxiv_context"] == []
        assert isinstance(result["messages"][-1], AIMessage)







# initialization tests
# def test_initialization():
#     retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#     assert retriever.max_docs_to_load == 5
#     assert retriever.chunk_size == 400
#     assert retriever.chunk_overlap == 100
#     assert retriever.k_results == 3

# def test_chroma_vector_store_creation():
#     with patch('arxiv_retriever.Chroma') as mock_chroma:
#         retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#         mock_chroma.assert_called_with(
#             collection_name='default',
#             embedding_function=retriever._embeddings.embed_query,
#             persist_directory='.',
#             client_settings=None
#         )

# functionality tests
# def test_document_loading():
#     with patch('arxiv_retriever.ArxivLoader') as mock_loader:
#         mock_documents, _ = process_items_safely(mock_loader.lazy_load())
#         retriever = ArxivContextRetriever(max_docs_to_load=2, chunk_size=400, chunk_overlap=100, k_results=3)
#         documents, _ = retriever('Research Idea')
#         assert len(documents) <= 5

# # edge case tests
# def test_empty_research_idea():
#     retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#     result = retriever(
#         state={
#             "main_ideas": ["monte carlo", "numerical analysis"],
#             "arxiv_context": [],
#             "messages": []
#         }
#     )
#     assert 'arxiv_context' in result and len(result['arxiv_context']) == 0

# def test_max_document_load():
#     with patch('arxiv_retriever.ArxivLoader') as mock_loader:
#         mock_documents, _ = process_items_safely(mock_loader.lazy_load())
#         retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#         documents, _ = retriever._load_documents('Research Idea with More Than Five Documents')
#         assert len(documents) == 5

# integration tests
# def test_end_to_end_retrieval():
#     with patch('arxiv_retriever.ArxivLoader') as mock_loader:
#         mock_documents, _ = process_items_safely(mock_loader.lazy_load())
#         retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#         result = retriever('Research Idea')
#         assert 'arxiv_context' in result
#         assert len(result['arxiv_context']) > 0

# def test_similarity_search_insufficient_results():
#     with patch('arxiv_retriever.ArxivLoader') as mock_loader:
#         mock_documents, _ = process_items_safely(mock_loader.lazy_load())
#         retriever = ArxivContextRetriever(max_docs_to_load=5, chunk_size=400, chunk_overlap=100, k_results=3)
#         result = retriever('Research Idea with Few Relevant Documents')
#         assert len(result['arxiv_context']) <= 3
