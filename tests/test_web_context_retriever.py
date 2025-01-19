import pytest
from langchain_core.messages.ai import AIMessage
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
        urls=urls,
        chunk_size=400,
        chunk_overlap=100,
        k_results=2
    )

class TestWebContextRetriever:

    def test_initialization(self, retriever):
        assert len(retriever._urls) == 3
        assert retriever._k_results == 2


    def test_empty_initialization(self):
        retriever = WebContextRetriever(
            urls=[],
            chunk_size=400,
            chunk_overlap=100,
            k_results=2
        )
        assert len(retriever._urls) == 0

    def test_web_context_retrieval_success(self, urls, mock_state):
        retriever = WebContextRetriever(
            urls=urls,
        )
        result = retriever(mock_state)
        assert result["calling_agent"] == WebContextRetriever.NAME
        assert len(result["web_context"]) > 0

    def test_web_context_retrieval_failure(self, mock_state):
        retriever = WebContextRetriever(
            urls=[],
            chunk_size=400,
            chunk_overlap=100,
            k_results=2
        )
        result = retriever(mock_state)
        assert result["calling_agent"] == WebContextRetriever.NAME
        assert len(result["web_context"]) == 0
        ai_message = result["messages"][-1]
        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "WEB CONTEXT RETRIEVAL: FAILED"
