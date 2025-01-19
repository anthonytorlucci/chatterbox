# test_research_context_supervisor.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from chatterbox.nodes.research_context_supervisor import ResearchContextSupervisor

@pytest.fixture
def base_state():
    """Base state fixture with empty contexts and a human message."""
    return {
        "messages": [HumanMessage(content="Test query")],
        "pdf_context": [],
        "web_context": [],
        "arxiv_context": [],
        "vdbs_context": [],
        "requires_pdf_context": True,
        "requires_web_context": True,
        "requires_arxiv_context": True,
        "requires_vdbs_context": True,
    }

class TestResearchContextSupervisor:

    def test_initialization(self):
        """Test proper initialization of ResearchContextSupervisor."""
        supervisor = ResearchContextSupervisor(
            has_pdf_paths=True,
            has_vector_dbs=True,
            has_urls=True,
            use_arxiv_search=True,
        )
        assert supervisor._has_pdf_paths is True
        assert supervisor._has_vector_dbs is True
        assert supervisor._has_urls is True
        assert supervisor._use_arxiv_search is True

        supervisor = ResearchContextSupervisor()
        assert supervisor._has_pdf_paths is False
        assert supervisor._has_vector_dbs is False
        assert supervisor._has_urls is False
        assert supervisor._use_arxiv_search is False

    def test_supervisor_name(self):
        """Test if the supervisor has the correct NAME attribute."""
        supervisor = ResearchContextSupervisor()
        assert supervisor.NAME == "research_context_supervisor"

    def test_empty_context_state(self, base_state):
        """Test supervisor behavior with empty context state, no pdf files, and no vector dbs."""
        supervisor = ResearchContextSupervisor(
            has_pdf_paths=False,
            has_vector_dbs=False,
            has_urls=False,
            use_arxiv_search=False,
        )
        result = supervisor(base_state)

        assert result["calling_agent"] == "research_context_supervisor"
        assert isinstance(result["messages"][-1], AIMessage)
        assert result.get("requires_pdf_context") == False
        assert result.get("requires_web_context") == False
        assert result.get("requires_arxiv_context") == False
        assert result.get("requires_vdbs_context") == False

    def test_partial_context_state(self, base_state):
        """Test supervisor behavior with some existing context."""
        base_state["pdf_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]
        base_state["web_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]

        supervisor = ResearchContextSupervisor(
            has_pdf_paths=True,
            has_vector_dbs=False,
            has_urls=False,
            use_arxiv_search=False
        )
        result = supervisor(base_state)

        assert result["calling_agent"] == "research_context_supervisor"
        assert isinstance(result["messages"][-1], AIMessage)
        assert result.get("requires_pdf_context") == False  # has_pdf_paths is True, however pdf_context is not empty
        assert result.get("requires_web_context") == False  # has_urls is False
        assert result.get("requires_arxiv_context") == False  # use_arxiv_search is False
        assert result.get("requires_vdbs_context") == False  # has_vector_dbs is False

    def test_full_context_state(self, base_state):
        """Test supervisor behavior when all contexts are present."""
        base_state["pdf_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]
        base_state["web_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]
        base_state["arxiv_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]
        base_state["vdbs_context"] = [
            Document(
                page_content="Example context",
                metadata={"source": "https://example.org"}
            ),
        ]

        supervisor = ResearchContextSupervisor(has_pdf_paths=True, has_vector_dbs=True)
        result = supervisor(base_state)

        assert result["calling_agent"] == "research_context_supervisor"
        assert isinstance(result["messages"][-1], AIMessage)
        assert result.get("requires_pdf_context") == False
        assert result.get("requires_web_context") == False
        assert result.get("requires_arxiv_context") == False
        assert result.get("requires_vdbs_context") == False


    def test_state_preservation(self, base_state):
        """Test that the original messages are preserved in the state."""
        original_messages = base_state["messages"].copy()
        supervisor = ResearchContextSupervisor()
        result = supervisor(base_state)

        assert len(result["messages"]) == len(original_messages) + 1
        assert result["messages"][:-1] == original_messages

    @pytest.mark.parametrize("has_pdf_paths,has_vector_dbs", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_different_configurations(self, base_state, has_pdf_paths, has_vector_dbs):
        """Test supervisor behavior with different configuration combinations."""
        supervisor = ResearchContextSupervisor(
            has_pdf_paths=has_pdf_paths,
            has_vector_dbs=has_vector_dbs
        )
        result = supervisor(base_state)

        assert result["calling_agent"] == "research_context_supervisor"
        assert isinstance(result["messages"][-1], AIMessage)
