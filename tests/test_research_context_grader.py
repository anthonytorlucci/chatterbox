import os
from unittest.mock import patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from chatterbox.language_models import (
    LargeLanguageModelConfig,
    LargeLanguageModelsEnum,
    #get_llm_model
)
from chatterbox.nodes.pdf_context_retriever import PdfContextRetriever
from chatterbox.nodes.web_context_retriever import WebContextRetriever
from chatterbox.nodes.arxiv_context_retriever import ArxivContextRetriever
from chatterbox.nodes.vdbs_context_retriever import VdbsContextRetriever
from chatterbox.nodes.research_context_grader import (
    ResearchContextGrader,
    #GradeDocuments
)

# ---- environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found.")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found.")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found.")

# Mock classes and functions
# class MockLLM:
#     def with_structured_output(self, output_class):
#         self.output_class = output_class
#         return self

#     def invoke(self, inputs):
#         return self.output_class(binary_score="yes")

@pytest.fixture
def mock_model_config():
    return LargeLanguageModelConfig(
        id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
        api_key=str(OPENAI_API_KEY),
        temperature=0.0,
        max_tokens=2000,
    )

@pytest.fixture
def mock_grader(mock_model_config):
    return ResearchContextGrader(model_config=mock_model_config)

@pytest.fixture
def mock_state():
    return {
        "research_prompt": "What is the impact of melting Artic ice on polar bears?",
        "calling_agent": PdfContextRetriever.NAME,
        "pdf_context": [
            # TODO: List of Documents
            Document(
                page_content="The consensus is clear – as Arctic sea ice melts, polar bears are finding it harder to hunt, mate and breed. While polar bears have shown some ability to adapt to changes in their surroundings – for example, by foraging for food on land, or swimming more to hunt for prey – scientists project that as sea ice diminishes, polar bears will find it harder to survive and populations will decline.",
                metadata={"source": "https://interactive.carbonbrief.org/polar-bears-climate-change-what-does-science-say/index.html"}
            ),
            Document(
                page_content="Polar bears are dependent upon Arctic sea ice for survival, traveling hundreds of miles across this critical habitat, hunting for prey and building snow cave dens to raise their cubs. More than 96 percent of the polar bear’s critical habitat is sea ice, and just four percent is onshore.",
                metadata={"source": "https://defenders.org/blog/2022/11/polar-bears-affected-climate-change"}
            ),
        ],
        "messages": []
    }

@pytest.fixture
def sample_document():
    """Fixture for creating a sample document"""
    return Document(
        page_content="AI and machine learning are transforming technology",
        metadata={"source": "test.pdf"}
    )

@pytest.fixture
def base_state():
    """Fixture for creating a base state dictionary"""
    return {
        "research_prompt": "What are the impacts of AI on technology?",
        "messages": []
    }

class TestResearchContextGrader:

    def test_research_context_grader_call_with_relevant_documents(self, mock_grader, mock_state):
        result = mock_grader(mock_state)
        assert len(result["pdf_context"]) == 2
        assert result["messages"][-1].content == "2 relevant documents found."

    def test_research_context_grader_call_with_no_documents(self, mock_grader, mock_state):
        mock_state["pdf_context"] = []
        result = mock_grader(mock_state)
        assert len(result["pdf_context"]) == 0
        assert result["messages"][-1].content == "No relevant documents found."

    def test_research_context_grader_call_with_unknown_agent(self, mock_grader, mock_state):
        mock_state["calling_agent"] = "unknown_agent"
        with pytest.raises(ValueError) as excinfo:
            mock_grader(mock_state)
        assert str(excinfo.value) == "Unknown calling agent: unknown_agent"

    def test_initialization(self, mock_model_config):
        """Test if ResearchContextGrader initializes correctly"""
        grader = ResearchContextGrader(model_config=mock_model_config)
        assert grader.NAME == "research_context_grader"
        assert hasattr(grader, 'pdf_context_grader')

    @pytest.mark.parametrize("calling_agent,context_key", [
        (PdfContextRetriever.NAME, "pdf_context"),
        (WebContextRetriever.NAME, "web_context"),
        (ArxivContextRetriever.NAME, "arxiv_context"),
        (VdbsContextRetriever.NAME, "vdbs_context"),
    ])
    def test_grader_with_relevant_document(self, mock_grader, sample_document, base_state, calling_agent, context_key):
        """Test grading with relevant documents for different retrievers"""
        state = base_state.copy()
        state["calling_agent"] = calling_agent
        state[context_key] = [sample_document]

        result = mock_grader(state)

        assert result["calling_agent"] == calling_agent
        assert len(result[context_key]) > 0
        assert isinstance(result["messages"][-1], AIMessage)
        assert "relevant documents found" in result["messages"][-1].content

    @pytest.mark.parametrize("calling_agent,context_key", [
        (PdfContextRetriever.NAME, "pdf_context"),
        (WebContextRetriever.NAME, "web_context"),
        (ArxivContextRetriever.NAME, "arxiv_context"),
        (VdbsContextRetriever.NAME, "vdbs_context"),
    ])
    def test_grader_with_irrelevant_document(self, mock_grader, base_state, calling_agent, context_key):
        """Test grading with irrelevant documents for different retrievers"""
        irrelevant_doc = Document(
            page_content="Recipe for chocolate cake",
            metadata={"source": "recipes.pdf"}
        )

        state = base_state.copy()
        state["calling_agent"] = calling_agent
        state[context_key] = [irrelevant_doc]

        result = mock_grader(state)

        assert result["calling_agent"] == calling_agent
        assert len(result[context_key]) == 0
        assert isinstance(result["messages"][-1], AIMessage)
        assert "No relevant documents found" in result["messages"][-1].content

    def test_grader_with_empty_documents(self, mock_grader, base_state):
        """Test grading with empty document list"""
        state = base_state.copy()
        state["calling_agent"] = PdfContextRetriever.NAME
        state["pdf_context"] = []

        result = mock_grader(state)

        assert result["calling_agent"] == PdfContextRetriever.NAME
        assert len(result["pdf_context"]) == 0
        assert isinstance(result["messages"][-1], AIMessage)
        assert "No relevant documents found" in result["messages"][-1].content

    def test_grader_with_invalid_calling_agent(self, mock_grader, base_state):
        """Test grading with invalid calling agent"""
        state = base_state.copy()
        state["calling_agent"] = "invalid_agent"

        with pytest.raises(ValueError) as exc_info:
            mock_grader(state)
        assert "Unknown calling agent" in str(exc_info.value)

    def test_grader_without_research_prompt(self, mock_grader, sample_document):
        """Test grading without research prompt"""
        state = {
            "calling_agent": PdfContextRetriever.NAME,
            "pdf_context": [sample_document],
            "messages": []
        }

        result = mock_grader(state)

        assert result["calling_agent"] == PdfContextRetriever.NAME
        assert len(result["pdf_context"]) == 0
        assert isinstance(result["messages"][-1], AIMessage)
        assert "No relevant documents found" in result["messages"][-1].content
