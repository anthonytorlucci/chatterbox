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

import dataclasses
import pytest
from chatterbox.language_models import LargeLanguageModelsAPIInfo

def test_large_language_models_api_info_valid_initialization():
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

def test_large_language_models_api_info_optional_urls():
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

def test_large_language_models_api_info_invalid_urls():
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

def test_large_language_models_api_info_immutability():
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

def test_large_language_models_api_info_required_fields():
    """Test that all required fields must be provided."""
    with pytest.raises(TypeError):
        LargeLanguageModelsAPIInfo(
            company="TestCompany"  # Missing required fields
        )

def test_large_language_models_api_info_empty_strings():
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

def test_large_language_models_api_info_equality():
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
