import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from chatterbox.nodes.triage import Triage, MainIdeas
from chatterbox.language_models import LargeLanguageModelConfig, LargeLanguageModelsEnum

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

# Fixtures
@pytest.fixture
def model_config_openai_gpt4o_mini():
    return LargeLanguageModelConfig(
        id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
        api_key=str(OPENAI_API_KEY),
        temperature=0.0,
        max_tokens=1000,
    )

@pytest.fixture
def model_config_openai_claude_haiku_3():
    return LargeLanguageModelConfig(
        id=LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_3_HAIKU,
        api_key=str(ANTHROPIC_API_KEY),
        temperature=0.0,
        max_tokens=1000,
    )

@pytest.fixture
def triage_agent(model_config_openai_gpt4o_mini):
    return Triage(model_config=model_config_openai_gpt4o_mini)

@pytest.fixture
def basic_state():
    return {
        "messages": [
            HumanMessage(content="Explain the impact of climate change on biodiversity")
        ]
    }

class TestMainIdeas:
    def test_main_ideas_model(self):
        """Test the MainIdeas pydantic model"""
        ideas = ["idea1", "idea2", "idea3"]
        main_ideas = MainIdeas(main_ideas=ideas)
        assert isinstance(main_ideas, BaseModel)
        assert main_ideas.main_ideas == ideas
        assert len(main_ideas.main_ideas) == 3

class TestTriage:
    # Test cases
    def test_triage_initialization(self, model_config_openai_gpt4o_mini):
        """Test proper initialization of Triage agent"""
        triage = Triage(
            model_config=model_config_openai_gpt4o_mini,
            max_num_main_ideas=5
        )
        assert triage.NAME == "triage"
        assert triage._max_num_main_ideas == 5
        assert hasattr(triage, 'main_ideas_extractor')

    def test_triage_custom_max_ideas(self, model_config_openai_gpt4o_mini):
        """Test initialization with custom max_num_main_ideas"""
        triage = Triage(model_config=model_config_openai_gpt4o_mini, max_num_main_ideas=3)
        assert triage._max_num_main_ideas == 3

    def test_triage_basic_extraction(self, triage_agent, basic_state):
        """Test basic extraction functionality"""
        result = triage_agent(basic_state)

        assert isinstance(result, dict)
        assert "calling_agent" in result
        assert "main_ideas" in result
        assert "research_prompt" in result
        assert "messages" in result
        assert "recursion_limit" in result

        assert result["calling_agent"] == "triage"
        assert isinstance(result["main_ideas"], list)
        assert len(result["main_ideas"]) <= 5  # default max number of ideas
        assert isinstance(result["messages"][-1], AIMessage)

    def test_triage_empty_state(self):
        """Test handling of empty state"""
        triage = Triage(
            model_config=LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B,
                api_key="",
                temperature=0.0,
                max_tokens=1024,
            )
        )
        with pytest.raises(ValueError, match="No messages found."):
            triage({})

    def test_triage_empty_messages(self):
        """Test handling of state with empty messages list"""
        triage = Triage(
            model_config=LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B,
                api_key="",
                temperature=0.0,
                max_tokens=1024,
            )
        )
        with pytest.raises(ValueError, match="No messages found."):
            triage({"messages": []})

    def test_triage_respects_max_ideas(self, model_config_openai_gpt4o_mini):
        """Test that the number of extracted ideas doesn't exceed max_num_main_ideas"""
        triage = Triage(model_config=model_config_openai_gpt4o_mini, max_num_main_ideas=3)
        state = {
            "messages": [
                HumanMessage(content="Explain quantum computing, cryptography, artificial intelligence, machine learning, and neural networks")
            ]
        }
        result = triage(state)
        assert len(result["main_ideas"]) <= 3

    def test_triage_output_format(self, triage_agent, basic_state):
        """Test the format of the output message"""
        result = triage_agent(basic_state)
        assert result["messages"][-1].content.startswith("Extracted main ideas: ")

    @pytest.mark.integration
    def test_triage_complex_prompt(self, triage_agent):
        """Integration test with a complex research prompt"""
        state = {
            "messages": [
                HumanMessage(content="""
                Analyze the intersection of quantum computing and artificial intelligence,
                specifically focusing on how quantum machine learning algorithms might
                revolutionize deep learning and neural network training.
                """)
            ]
        }
        result = triage_agent(state)
        assert len(result["main_ideas"]) > 0
        assert isinstance(result["main_ideas"], list)
        assert all(isinstance(idea, str) for idea in result["main_ideas"])
