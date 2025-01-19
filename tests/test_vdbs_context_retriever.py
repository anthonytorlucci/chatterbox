from pathlib import Path
import pytest
from chatterbox.nodes.vdbs_context_retriever import VdbsContextRetriever


@pytest.fixture
def mock_state():
    return {
        "main_ideas": ["machine learning", "artificial intelligence"],
        "vdbs_context": [],
        "messages": []
    }

@pytest.fixture
def collection_names():
    return [
        "medium_generative_driven_design",
        "medium_neural_networks_are_fundamentally_bayesian"
    ]

@pytest.fixture
def retriever(collection_names):
    return VdbsContextRetriever(
        db_dir=Path(__file__).parent.parent.joinpath("vdbs_documents", "chroma_research_notes_ollama_emb_db"),
        collection_names=collection_names,
        k_results=2,
    )

class TestWebContextRetriever:

    def test_initialization(self, retriever):
        assert len(retriever._collection_names) == 2
        assert retriever._k_results == 2

    def test_web_context_retrieval_success(self, collection_names, mock_state):
        retriever = VdbsContextRetriever(
            db_dir=Path(__file__).parent.parent.joinpath("vdbs_documents", "chroma_research_notes_ollama_emb_db"),
            collection_names=collection_names,
            k_results=2,
        )
        result = retriever(mock_state)
        assert result["calling_agent"] == VdbsContextRetriever.NAME
        assert len(result["vdbs_context"]) > 0
