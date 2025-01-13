"""
These tests cover:

1. **Type checking**: Verifies that all fields have the correct types
2. **Optional fields**: Tests handling of optional `urls` parameter
3. **Invalid input**: Tests that invalid input (like non-list urls) raises appropriate errors
4. **Immutability**: Verifies that the dataclass is frozen and fields can't be modified
5. **Required fields**: Tests that all required fields must be provided
6. **Empty values**: Tests handling of empty strings and lists
7. **Equality**: Tests that two instances with the same values are considered equal

Additional tests you might consider adding:

1. Test with very long strings for description
2. Test with special characters in strings
3. Test with different URL formats
4. Test serialization/deserialization if needed
5. Test with Unicode characters
6. Test with maximum/minimum values if applicable
"""

import os
import dataclasses
import pytest
from chatterbox.language_models import (
    LargeLanguageModelsAPIInfo,
    LargeLanguageModelsEnumInterface,
    LargeLanguageModelsEnum,
    LargeLanguageModelConfig,
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

# ---- tests
class TestLargeLanguageModelsAPIInfo:

    def test_large_language_models_api_info_valid_initialization(self):
        """Test valid initialization of LargeLanguageModelsAPIInfo."""
        info = LargeLanguageModelsAPIInfo(
            company="TestCompany",
            generic_name="test-model",
            api_reference="test-ref",
            description="A test model",
            has_tools=True,
            urls=["https://test.com"]
        )

        assert isinstance(info.company, str)
        assert isinstance(info.generic_name, str)
        assert isinstance(info.api_reference, str)
        assert isinstance(info.description, str)
        assert isinstance(info.has_tools, bool)
        assert isinstance(info.urls, list)
        assert all(isinstance(url, str) for url in info.urls)

    def test_large_language_models_api_info_optional_urls(self):
        """Test initialization with urls=None is valid."""
        info = LargeLanguageModelsAPIInfo(
            company="TestCompany",
            generic_name="test-model",
            api_reference="test-ref",
            description="A test model",
            has_tools=False,
            urls=None
        )

        assert info.urls is None

    def test_large_language_models_api_info_invalid_urls(self):
        """Test that invalid urls parameter raises ValueError."""
        with pytest.raises(ValueError):
            LargeLanguageModelsAPIInfo(
                company="TestCompany",
                generic_name="test-model",
                api_reference="test-ref",
                description="A test model",
                has_tools=True,
                urls="not-a-list"  # This should raise ValueError
            )

    def test_large_language_models_api_info_immutability(self):
        """Test that the dataclass is immutable (frozen=True)."""
        info = LargeLanguageModelsAPIInfo(
            company="TestCompany",
            generic_name="test-model",
            api_reference="test-ref",
            description="A test model",
            has_tools=True,
            urls=["https://test.com"]
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            info.company = "NewCompany"

    def test_large_language_models_api_info_required_fields(self):
        """Test that all required fields must be provided."""
        with pytest.raises(TypeError):
            LargeLanguageModelsAPIInfo(
                company="TestCompany"  # Missing required fields
            )

    def test_large_language_models_api_info_empty_strings(self):
        """Test initialization with empty strings."""
        info = LargeLanguageModelsAPIInfo(
            company="",
            generic_name="",
            api_reference="",
            description="",
            has_tools=False,
            urls=[]
        )

        assert info.company == ""
        assert info.generic_name == ""
        assert info.api_reference == ""
        assert info.description == ""
        assert info.urls == []

    def test_large_language_models_api_info_equality(self):
        """Test equality comparison of two instances."""
        info1 = LargeLanguageModelsAPIInfo(
            company="TestCompany",
            generic_name="test-model",
            api_reference="test-ref",
            description="A test model",
            has_tools=True,
            urls=["https://test.com"]
        )

        info2 = LargeLanguageModelsAPIInfo(
            company="TestCompany",
            generic_name="test-model",
            api_reference="test-ref",
            description="A test model",
            has_tools=True,
            urls=["https://test.com"]
        )

        assert info1 == info2

class TestLargeLanguageModelsEnumInterface:

    def test_enum_interface_properties(self):
        # Test that the interface properties are correctly defined
        assert hasattr(LargeLanguageModelsEnumInterface, 'company')
        assert hasattr(LargeLanguageModelsEnumInterface, 'generic_name')
        assert hasattr(LargeLanguageModelsEnumInterface, 'api_reference')
        assert hasattr(LargeLanguageModelsEnumInterface, 'description')
        assert hasattr(LargeLanguageModelsEnumInterface, 'has_tools')
        assert hasattr(LargeLanguageModelsEnumInterface, 'urls')

class TestLargeLanguageModelsEnum:

    def test_enum_members_exist(self):
        # Test that the enum members are correctly defined
        assert hasattr(LargeLanguageModelsEnum, 'OLLAMA_LLAMA_32_3B')
        assert hasattr(LargeLanguageModelsEnum, 'OLLAMA_MARCO_01_7B')
        assert hasattr(LargeLanguageModelsEnum, 'OLLAMA_FALCON3_7B')

    def test_enum_member_properties(self):
        # Test that the properties of each enum member are correctly defined
        model = LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B
        assert model.company == "Ollama"
        assert model.generic_name == "ollama llama3.2"
        assert model.api_reference == "llama3.2"
        #assert model.description == "Meta's Llama 3.2 goes small with 1B and 3B models."
        assert model.has_tools is True
        assert model.urls == ["https://ollama.com/library/llama3.2"]

        model = LargeLanguageModelsEnum.OLLAMA_FALCON3_7B
        assert model.company == "Ollama"
        assert model.generic_name == "ollama falcon3"
        assert model.api_reference == "falcon3"
        #assert model.description == "A family of efficient AI models under 10B parameters performant in science, math, and coding through innovative training techniques."
        assert model.has_tools is False
        assert model.urls == ["https://ollama.com/library/falcon3"]

    def test_enum_member_inheritance(self):
        # Test that the enum members inherit from the interface
        model = LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B
        assert isinstance(model, LargeLanguageModelsEnumInterface)

        model = LargeLanguageModelsEnum.OLLAMA_FALCON3_7B
        assert isinstance(model, LargeLanguageModelsEnumInterface)



class TestLargeLanguageModelConfig:

    def test_large_language_model_config_instantiation(self):
        config = LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O,
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000,
            timeout=30.0
        )
        assert config.id == LargeLanguageModelsEnum.OPENAI_GPT_4O
        assert config.api_key == "your-api-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.max_retries == 3
        assert config.timeout == 30.0

    def test_large_language_model_config_default_values(self):
        config = LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_3_HAIKU,
            api_key=str(ANTHROPIC_API_KEY),
            temperature=0.5,
            max_tokens=500
        )
        assert config.max_retries == 3
        assert config.timeout is None

    def test_large_language_model_config_invalid_temperature(self):
        with pytest.raises(ValueError):
            LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.FIREWORKS_MIXTRAL_MOE_8X22B_INSTRUCT,
                api_key=str(FIREWORKS_API_KEY),
                temperature=2.0,  # Invalid temperature value
                max_tokens=1000
            )

    def test_large_language_model_config_invalid_max_tokens(self):
        with pytest.raises(ValueError):
            LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B,
                api_key="",
                temperature=0.5,
                max_tokens=-100  # Invalid max_tokens value
            )

    def test_large_language_model_config_invalid_max_retries(self):
        with pytest.raises(ValueError):
            LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
                api_key="your-api-key",
                temperature=0.5,
                max_tokens=1000,
                max_retries=-1  # Invalid max_retries value
            )

    def test_large_language_model_config_invalid_timeout(self):
        with pytest.raises(ValueError):
            LargeLanguageModelConfig(
                id=LargeLanguageModelsEnum.OLLAMA_PHI4_14B,
                api_key="",
                temperature=0.5,
                max_tokens=1000,
                timeout=-10.0  # Invalid timeout value
            )

    def test_large_language_model_config_as_dict(self):
        config = LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O,
            api_key=str(OPENAI_API_KEY),
            temperature=0.7,
            max_tokens=1000,
            timeout=30.0
        )
        config_dict = dataclasses.asdict(config)
        assert config_dict == {
            'id': LargeLanguageModelsEnum.OPENAI_GPT_4O,
            'api_key': OPENAI_API_KEY,
            'temperature': 0.7,
            'max_tokens': 1000,
            'max_retries': 3,
            'timeout': 30.0
        }
