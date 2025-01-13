import logging
import os
import pytest
from langchain_core.documents import Document
from chatterbox.language_models import (
    LargeLanguageModelConfig,
    LargeLanguageModelsEnum
)
from chatterbox.nodes.research_context_summarizer import ResearchContextSummarizer

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

@pytest.fixture
def ai_research_state():
    return {
        "research_prompt": "Summarize research on AI.",
        "pdf_context": [],
        "web_context": [
            Document(
                page_content="""1. AI beats humans on some tasks, but not on all.

AI has surpassed human performance on several benchmarks, including some in image classification, visual reasoning, and English understanding. Yet it trails behind on more complex tasks like competition-level mathematics, visual commonsense reasoning and planning.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""2. Industry continues to dominate frontier AI research.

In 2023, industry produced 51 notable machine learning models, while academia contributed only 15. There were also 21 notable models resulting from industry-academia collaborations in 2023, a new high.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""3. Frontier models get way more expensive.

According to AI Index estimates, the training costs of state-of-the-art AI models have reached unprecedented levels. For example, OpenAI’s GPT-4 used an estimated $78 million worth of compute to train, while Google’s Gemini Ultra cost $191 million for compute.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""4. The United States leads China, the EU, and the U.K. as the leading source of top AI models.

In 2023, 61 notable AI models originated from U.S.-based institutions, far outpacing the European Union’s 21 and China’s 15.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""5. Robust and standardized evaluations for LLM responsibility are seriously lacking.

New research from the AI Index reveals a significant lack of standardization in responsible AI reporting. Leading developers, including OpenAI, Google, and Anthropic, primarily test their models against different responsible AI benchmarks. This practice complicates efforts to systematically compare the risks and limitations of top AI models.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""6. Generative AI investment skyrockets.

Despite a decline in overall AI private investment last year, funding for generative AI surged, nearly octupling from 2022 to reach $25.2 billion. Major players in the generative AI space, including OpenAI, Anthropic, Hugging Face, and Inflection, reported substantial fundraising rounds.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""7. The data is in: AI makes workers more productive and leads to higher quality work.

In 2023, several studies assessed AI’s impact on labor, suggesting that AI enables workers to complete tasks more quickly and to improve the quality of their output. These studies also demonstrated AI’s potential to bridge the skill gap between low- and high-skilled workers. Still other studies caution that using AI without proper oversight can lead to diminished performance.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""8. Scientific progress accelerates even further, thanks to AI.

In 2022, AI began to advance scientific discovery. 2023, however, saw the launch of even more significant science-related AI applications—from AlphaDev, which makes algorithmic sorting more efficient, to GNoME, which facilitates the process of materials discovery.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""9. The number of AI regulations in the United States sharply increases.

The number of AI-related regulations in the U.S. has risen significantly in the past year and over the last five years. In 2023, there were 25 AI-related regulations, up from just one in 2016. Last year alone, the total number of AI-related regulations grew by 56.3%.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
            Document(
                page_content="""10. People across the globe are more cognizant of AI’s potential impact—and more nervous.

A survey from Ipsos shows that, over the last year, the proportion of those who think AI will dramatically affect their lives in the next three to five years has increased from 60% to 66%. Moreover, 52% express nervousness toward AI products and services, marking a 13 percentage point rise from 2022. In America, Pew data suggests that 52% of Americans report feeling more concerned than excited about AI, rising from 38% in 2022.""",
                metadata={"source": "https://aiindex.stanford.edu/report/"}
            ),
        ],
        "arxiv_context": [],
        "vdbs_context": [],
        "messages": []
    }

@pytest.fixture
def model_config():
    return LargeLanguageModelConfig(
        id=LargeLanguageModelsEnum.OLLAMA_PHI4_14B,
        api_key="",
        temperature=0.0,
        max_tokens=2000,
    )

@pytest.fixture
def summarizer(model_config):
    return ResearchContextSummarizer(model_config)

@pytest.fixture
def summarizer_gpt4o_mini(model_config):
    return ResearchContextSummarizer(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
            api_key=str(OPENAI_API_KEY),
            temperature=0.0,
            max_tokens=2000,
        )
    )

class TestResearchContextSummarizer:
    pass
    # initialization tests
    def test_initialization(self, summarizer):
        """Ensure that the `ResearchContextSummarizer` initializes correctly."""
        assert summarizer.NAME == "research_context_summarizer"


    def test_summarize_with_valid_input(self, summarizer_gpt4o_mini, ai_research_state):
        """Verify that summarizer processes valid input correctly."""
        result = summarizer_gpt4o_mini(ai_research_state)
        assert "summarized_context" in result
        assert isinstance(result["summarized_context"], str)
        assert len(result["summarized_context"]) > 0


    def test_summarize_with_empty_contexts(self, summarizer):
        """Ensure that a `ValueError` is raised when all context lists are empty."""
        state = {
            "research_prompt": "Summarize research on AI.",
            "pdf_context": [],
            "web_context": [],
            "arxiv_context": [],
            "vdbs_context": [],
            "messages": []
        }
        with pytest.raises(ValueError, match="All context lists are empty or research prompt is empty."):
            summarizer(state)

    def test_summarize_with_empty_research_prompt(self, summarizer):
        """Ensure that a `ValueError` is raised when the research prompt is empty."""
        state = {
            "pdf_context": [MagicMock(page_content="Document 1 content")],
            "web_context": [],
            "arxiv_context": [],
            "vdbs_context": [],
            "messages": []
        }
        with pytest.raises(ValueError, match="All context lists are empty or research prompt is empty."):
            summarizer(state)

    def test_logging_during_summarization(self, summarizer_gpt4o_mini, ai_research_state, caplog):
        """Verify that logging occurs as expected during the summarization process."""
        with caplog.at_level(logging.INFO):
            summarizer_gpt4o_mini(ai_research_state)
            assert "FORMALIZE PROMPT" in caplog.text

# ---
