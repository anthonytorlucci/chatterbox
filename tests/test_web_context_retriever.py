import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages.ai import AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from chatterbox.researcher_interface import ResearcherInterface
from chatterbox.utils import process_items_safely
from chatterbox.nodes.web_context_retriever import WebContextRetriever


@pytest.fixture
def mock_state():
    return {
        "main_ideas": ["machine learning", "artificial intelligence"],
        "web_context": [],
        "messages": []
    }

@pytest.fixture
def urls():
    return [
        "https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning",
        "https://www.sas.com/en_us/insights/analytics/machine-learning.html",
        "https://aws.amazon.com/what-is/machine-learning/"
    ]

@pytest.fixture
def retriever(urls):
    return WebContextRetriever(
        urls=[],
        add_search=False,  # Start with search disabled for simplicity
        max_search_results=10,
        chunk_size=400,
        chunk_overlap=100,
        k_results=3
    )

@pytest.fixture
def retriever_with_urls(urls):
    return WebContextRetriever(
        urls=urls,
        add_search=False,  # Start with search disabled for simplicity
        max_search_results=10,
        chunk_size=400,
        chunk_overlap=100,
        k_results=3
    )

@pytest.fixture
def retriever_with_search(urls):
    return WebContextRetriever(
        urls=[],
        add_search=True,  # test adding search
        max_search_results=10,
        chunk_size=400,
        chunk_overlap=100,
        k_results=3
    )

class TestWebContextRetriever:

    def test_initialization(self, retriever):
        assert len(retriever._urls) == 0
        assert retriever._add_search == False
        assert retriever._k_results == 3
        assert retriever._max_search_results == 10

    def test_initialization_with_urls(self, retriever_with_urls):
        assert len(retriever_with_urls._urls) == 3
        assert retriever_with_urls._add_search == False
        assert retriever_with_urls._k_results == 3
        assert retriever_with_urls._max_search_results == 10

    def test_web_context_retriever_initialization_no_search(self, retriever):
        assert not retriever._add_search

    def test_web_context_retriever_initialization_with_search(self, retriever_with_search):
        assert retriever_with_search._add_search

    def test_web_context_retriever_default_parameters(self):
        retriever = WebContextRetriever(urls=[])
        assert isinstance(retriever._urls, list)
        assert len(retriever._urls) == 0
        assert retriever._add_search == False


    def test_web_context_retrieval_success(self, urls, mock_state):
        retriever = WebContextRetriever(
            urls=urls,
            add_search=False
        )
        result = retriever(mock_state)
        assert result["calling_agent"] == WebContextRetriever.NAME
        assert len(result["web_context"]) > 0
        assert not result["requires_web_context"]


# @patch('web_context_retriever.DuckDuckGoSearchAPIWrapper')
# def test_web_context_retrieval_empty_main_ideas(mock_search):
#     state = {
#         "main_ideas": [],
#         "messages": []
#     }
#     retriever = WebContextRetriever(add_search=True)

#     # Mock search results
#     mock_search.return_value.invoke.side_effect = [
#         [{"link": "url1"}, {"link": "url2"}],
#         [{"link": "url3"}, {"link": "url4"}]
#     ]

#     documents = []
#     for url in ["url1", "url2", "url3", "url4"]:
#         loader = WebBaseLoader(url)
#         docs, _ = process_items_safely(loader.lazy_load())
#         documents.extend(docs)

#     all_splits = retriever._text_splitter.split_documents(documents)
#     uuids = [str(uuid4()) for _ in range(len(all_splits))]
#     retriever._vector_store.add_documents.assert_called_with(
#         documents=all_splits, ids=uuids
#     )

#     # Mock similarity search
#     mock_vector_search = MagicMock()
#     mock_vector_search.return_value = []
#     retriever._vector_store.similarity_search_by_vector.side_effect = [mock_vector_search]

#     result = retriever(state)
#     assert result["calling_agent"] == WebContextRetriever.NAME
#     assert len(result["web_context"]) == 0
#     assert not result["requires_web_context"]
#     # Additional assertions as necessary

    # @patch('web_context_retriever.DuckDuckGoSearchAPIWrapper')
    # def test_web_context_retrieval_exception(mock_search):
    #     state = {
    #         "main_ideas": ["Test Idea"],
    #         "messages": []
    #     }
    #     retriever = WebContextRetriever(add_search=True)
    #
    #     # Mock search results
    #     mock_search.return_value.invoke.side_effect = Exception("RateLimitError")
    #
    #     documents = []
    #     for url in ["url1", "url2"]:
    #         loader = WebBaseLoader(url)
    #         docs, _ = process_items_safely(loader.lazy_load())
    #         documents.extend(docs)
    #
    #     all_splits = retriever._text_splitter.split_documents(documents)
    #     uuids = [str(uuid4()) for _ in range(len(all_splits))]
    #     retriever._vector_store.add_documents.assert_called_with(
    #         documents=all_splits, ids=uuids
    #     )
    #
    #     # Mock similarity search
    #     mock_vector_search = MagicMock()
    #     mock_vector_search.return_value = [
    #         {"page_content": "Test Content 1"},
    #         {"page_content": "Test Content 2"}
    #     ]
    #     retriever._vector_store.similarity_search_by_vector.side_effect = [mock_vector_search]
    #
    #     result = retriever(state)
    #     assert result["calling_agent"] == WebContextRetriever.NAME
    #     assert len(result["web_context"]) == 2
    #     assert not result["requires_web_context"]
    #     # Additional assertions as necessary

    # ---
    # ---
    # ---
    # @patch('chatterbox.nodes.web_context_retriever.DuckDuckGoSearchResults')
    # def test_add_search(self, mock_duckduckgo, retriever_with_search):
    #     mock_duckduckgo.return_value.invoke.return_value = [
    #         {"link": "https://example.com/search1"},
    #         {"link": "https://example.com/search2"}
    #     ]
    #
    #     state = {
    #         "main_ideas": ["machine learning"],
    #         "web_context": [],
    #         "messages": []
    #     }
    #
    #     result_state = retriever_with_search(state)
    #     assert len(retriever_with_search._urls) == 4
    #     assert "https://example.com/search1" in retriever_with_search._urls

    # def test_retrieve_web_context_success(retriever, mock_state):
    #     # Mock the vector store and embedding methods
    #     with patch.object(retriever._vector_store, 'similarity_search_by_vector', return_value=[MagicMock()]):
    #         with patch.object(retriever._embeddings, 'embed_query', return_value=MagicMock()):
    #             result = retriever(mock_state)
    #
    #     assert "WEB CONTEXT RETRIEVAL: SUCCESS" in [msg.content for msg in result["messages"]]
    #     assert result["requires_web_context"] is False

    # def test_retrieve_web_context_failure(retriever, mock_state):
    #     # Mock the vector store to raise an exception
    #     with patch.object(retriever._vector_store, 'similarity_search_by_vector', side_effect=Exception()):
    #         result = retriever(mock_state)
    #
    #     assert "WEB CONTEXT RETRIEVAL: FAILED" in [msg.content for msg in result["messages"]]
    #     assert result["requires_web_context"] is True

    # def test_k_results_adjustment(retriever, mock_state):
    #     # Adjust the number of documents to be less than k_results
    #     retriever._vector_store.add_documents = MagicMock()
    #     retriever._vector_store.similarity_search_by_vector.return_value = [MagicMock()] * 2
    #
    #     result = retriever(mock_state)
    #
    #     assert retriever._k_results == 2
    #
    # Additional tests can be added to cover more edge cases and scenarios





# # Mock classes and functions
# class MockWebBaseLoader(WebBaseLoader):
#     def lazy_load(self):
#         return [("Mock Document Content", {})]

# class MockOllamaEmbeddings(OllamaEmbeddings):
#     def embed_query(self, text):
#         return [0.1, 0.2, 0.3]

# class MockChroma(Chroma):
#     def add_documents(self, documents, ids):
#         pass

#     def similarity_search_by_vector(self, embedding, k):
#         return ["Mock Document"]

# class MockDuckDuckGoSearchAPIWrapper(DuckDuckGoSearchAPIWrapper):
#     def run(self, query):
#         return [{"link": "http://mocksearch.com"}]

# class MockDuckDuckGoSearchResults(DuckDuckGoSearchResults):
#     def invoke(self, idea):
#         return [{"link": "http://mocksearch.com"}]

# # Fixtures
# @pytest.fixture
# def mock_web_context_retriever():
#     with patch('web_context_retriever.WebBaseLoader', new=MockWebBaseLoader):
#         with patch('web_context_retriever.OllamaEmbeddings', new=MockOllamaEmbeddings):
#             with patch('web_context_retriever.Chroma', new=MockChroma):
#                 with patch('web_context_retriever.DuckDuckGoSearchAPIWrapper', new=MockDuckDuckGoSearchAPIWrapper):
#                     with patch('web_context_retriever.DuckDuckGoSearchResults', new=MockDuckDuckGoSearchResults):
#                         return WebContextRetriever(urls=["http://example.com"])

# # Tests
# def test_initialization(mock_web_context_retriever):
#     assert mock_web_context_retriever.NAME == "web_context_retriever"
#     assert mock_web_context_retriever._k_results == 3
#     assert mock_web_context_retriever._max_search_results == 10
#     assert mock_web_context_retriever._text_splitter.chunk_size == 400
#     assert mock_web_context_retriever._text_splitter.chunk_overlap == 100

# def test_add_search(mock_web_context_retriever):
#     mock_web_context_retriever._add_search = True
#     mock_web_context_retriever._urls = ["http://example.com"]
#     mock_web_context_retriever._main_ideas = ["test idea"]
#     mock_web_context_retriever.__call__({"main_ideas": ["test idea"]})
#     assert len(mock_web_context_retriever._urls) > 1

# def test_document_loading(mock_web_context_retriever):
#     documents = mock_web_context_retriever._load_documents(["http://example.com"])
#     assert len(documents) == 1
#     assert documents[0] == "Mock Document Content"

# def test_document_splitting(mock_web_context_retriever):
#     documents = ["Mock Document Content"]
#     splits = mock_web_context_retriever._text_splitter.split_documents(documents)
#     assert len(splits) == 1

# def test_vector_store_operations(mock_web_context_retriever):
#     documents = ["Mock Document Content"]
#     splits = mock_web_context_retriever._text_splitter.split_documents(documents)
#     uuids = [str(uuid4()) for _ in range(len(splits))]
#     mock_web_context_retriever._vector_store.add_documents(documents=splits, ids=uuids)
#     results = mock_web_context_retriever._vector_store.similarity_search_by_vector(embedding=[0.1, 0.2, 0.3], k=3)
#     assert len(results) == 1

# def test_call_with_main_ideas(mock_web_context_retriever):
#     state = {
#         "main_ideas": ["test idea"],
#         "messages": []
#     }
#     result = mock_web_context_retriever(state)
#     assert result["calling_agent"] == "web_context_retriever"
#     assert len(result["web_context"]) == 1
#     assert result["requires_web_context"] == False
#     assert len(result["messages"]) == 1
#     assert result["messages"][0].content == "WEB CONTEXT RETRIEVAL: SUCCESS"

# def test_call_without_main_ideas(mock_web_context_retriever):
#     state = {
#         "main_ideas": [],
#         "messages": []
#     }
#     result = mock_web_context_retriever(state)
#     assert result["calling_agent"] == "web_context_retriever"
#     assert len(result["web_context"]) == 0
#     assert result["requires_web_context"] == False
#     assert len(result["messages"]) == 1
#     assert result["messages"][0].content == "WEB CONTEXT RETRIEVAL: SUCCESS"

# def test_call_with_error(mock_web_context_retriever):
#     with patch.object(MockChroma, 'similarity_search_by_vector', side_effect=Exception("Mock Exception")):
#         state = {
#             "main_ideas": ["test idea"],
#             "messages": []
#         }
#         result = mock_web_context_retriever(state)
#         assert result["calling_agent"] == "web_context_retriever"
#         assert len(result["web_context"]) == 0
#         assert result["requires_web_context"] is None
#         assert len(result["messages"]) == 1
#         assert result["messages"][0].content == "WEB CONTEXT RETRIEVAL: FAILED"
