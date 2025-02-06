# standard lib
import enum
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional, List
# third party
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_fireworks import ChatFireworks
from langchain_ollama import ChatOllama
# local

# https://docs.anthropic.com/en/docs/about-claude/models
# https://platform.openai.com/docs/models/o1
@dataclass(frozen=True)
class LargeLanguageModelsAPIInfo:
    """Information about a Large Language Model API.

    Attributes:
        company: The company that provides the model
        generic_name: Common name for the model
        api_reference: Reference string used in API calls
        urls: Optional list of relevant URLs for documentation
        context_window: Maximum context window size
        training_cutoff: Date of training data cutoff
    """
    company: str
    generic_name: str
    api_reference: str
    description: str
    has_tools: bool
    urls: Optional[List[str]] = None

    def __post_init__(self):
        """Validate the data after initialization."""
        if self.urls and not isinstance(self.urls, list):
            raise ValueError("urls must be a list of strings")


class LargeLanguageModelsEnumInterface(enum.Enum):
    """
    An interface enum class that defines properties for large language models.

    This enum interface provides standardized access to common properties of language models
    such as company name, model name, API references, descriptions, and related URLs.

    Properties:
        company: The company that created/owns the language model
        generic_name: The general/common name of the language model
        api_reference: Reference information for the model's API
        description: A description of the language model
        urls: Related URLs for the language model
    """
    @property
    def company(self):
        return self.value.company

    @property
    def generic_name(self):
        return self.value.generic_name

    @property
    def api_reference(self):
        return self.value.api_reference

    @property
    def description(self):
        return self.value.description

    @property
    def has_tools(self):
        return self.value.has_tools

    @property
    def urls(self):
        return self.value.urls

# reference: [Chat models](https://python.langchain.com/docs/integrations/chat/)
# [OpenAI models](https://platform.openai.com/docs/models)
class LargeLanguageModelsEnum(LargeLanguageModelsEnumInterface):
    """
    An enumeration of supported large language models with their specifications.

    This enum inherits from LargeLanguageModelsEnumInterface and provides a comprehensive
    list of available language models across different providers including OpenAI,
    Anthropic, Fireworks, and Ollama. Each enum value contains detailed information about
    the model including:
        - Company/provider name
        - Generic/common name
        - API reference identifier
        - Detailed description
        - Related documentation URLs

    Models are grouped by provider and include various capabilities and sizes:
        - OpenAI: O1 series, GPT-4O series
        - Anthropic: Claude 3 and 3.5 series
        - Fireworks: Llama 3.x series, Mixtral, Zephyr, Qwen
        - Ollama: Llama, Marco, Falcon

    Example:
        model_info = LargeLanguageModelsEnum.OPENAI_O1_PREVIEW
        company = model_info.company  # Returns "OpenAI"
        api_ref = model_info.api_reference  # Returns "o1-preview"
    """
    OLLAMA_LLAMA_32_3B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama llama3.2",
        api_reference="llama3.2",
        description="""
            Meta's Llama 3.2 goes small with 1B and 3B models.""",
        has_tools=True,
        urls=["https://ollama.com/library/llama3.2"]
    )
    OLLAMA_MARCO_01_7B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama marco-01",
        api_reference="marco-o1",
        description="""
            An open large reasoning model for real-world solutions by the Alibaba
            International Digital Commerce Group (AIDC-AI).""",
        has_tools=False,
        urls=["https://ollama.com/library/marco-o1"]
    )
    OLLAMA_FALCON3_7B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama falcon3",
        api_reference="falcon3",
        description="""
            A family of efficient AI models under 10B parameters performant in
            science, math, and coding through innovative training techniques.
        """,
        has_tools=False,
        urls=["https://ollama.com/library/falcon3"]
    )
    OLLAMA_GRANITE_31_MOE_3B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama granite3.1-moe",
        api_reference="granite3.1-moe",
        description="""
            The IBM Granite 1B and 3B models are long-context mixture of
            experts (MoE) Granite models from IBM designed for low latency
            usage.
        """,
        has_tools=True,
        urls=["https://ollama.com/library/granite3.1-moe"]
    )
    OLLAMA_GRANITE_31_DENSE_8B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama granite3.1-dense",
        api_reference="granite3.1-dense",
        description="""
            The IBM Granite 2B and 8B models are text-only dense LLMs trained
            on over 12 trillion tokens of data, demonstrated significant
            improvements over their predecessors in performance and speed in
            IBM’s initial testing.
        """,
        has_tools=True,
        urls=["https://ollama.com/library/granite3.1-dense"]
    )
    OLLAMA_PHI4_14B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama phi4",
        api_reference="phi4",
        description="""
        Phi 4 is a 14B parameter, state-of-the-art open model from Microsoft.
        """,
        has_tools=False,
        urls=["https://ollama.com/library/phi4"]
    )
    OLLAMA_DEEPSEEK_R1_32B = LargeLanguageModelsAPIInfo(
        company="Ollama",
        generic_name="ollama deepseek-r1 32b",
        api_reference="deepseek-r1:32b",
        description="""
        DeepSeek's first generation reasoning models with comparable performance to OpenAI-o1.
        """,
        has_tools=False,
        urls=["https://ollama.com/library/deepseek-r1:32b"]
    )
    # OLLAMA_ = LargeLanguageModelsAPIInfo(
    #     company="Ollama",
    #     generic_name="",
    #     api_reference="",
    #     description="""
    #     """,
    #     has_tools=False,
    #     urls=[""]
    # )
    OPENAI_GPT_4O = LargeLanguageModelsAPIInfo(
        company="OpenAI",
        generic_name="openai gpt-4o",
        api_reference="gpt-4o",
        description="""
            Our high-intelligence flagship model for complex, multi-step
            tasks. GPT-4o is cheaper and faster than GPT-4 Turbo. """,
        has_tools=True,
        urls=["https://platform.openai.com/docs/models"]
    )
    OPENAI_GPT_4O_MINI = LargeLanguageModelsAPIInfo(
        company="OpenAI",
        generic_name="openai gpt-4o-mini",
        api_reference="gpt-4o-mini",
        description="""
            Our affordable and intelligent small model for fast, lightweight
            tasks. GPT-4o mini is cheaper and more capable than GPT-3.5
            Turbo.""",
        has_tools=True,
        urls=["https://platform.openai.com/docs/models"]
    )
    ANTHROPIC_CLAUDE_35_SONNET = LargeLanguageModelsAPIInfo(
        company="Anthropic",
        generic_name="anthropic claude 3.5 sonnet",
        api_reference="claude-3-5-sonnet-20241022",
        description="""
        """,
        has_tools=True,
        urls=None
    )
    ANTHROPIC_CLAUDE_35_HAIKU = LargeLanguageModelsAPIInfo(
        company="Anthropic",
        generic_name="anthropic claude 3.5 haiku",
        api_reference="claude-3-5-haiku-20241022",
        description="""
        """,
        has_tools=True,
        urls=None
    )
    ANTHROPIC_CLAUDE_3_HAIKU = LargeLanguageModelsAPIInfo(
        company="Anthropic",
        generic_name="anthropic claude 3 haiku",
        api_reference="claude-3-haiku-20240307",
        description="""
        """,
        has_tools=True,
        urls=None
    )
    FIREWORKS_LLAMA_31_405B_INSTRUCT = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks llama 3.1 405B instruct",
        api_reference="accounts/fireworks/models/llama-v3p1-405b-instruct",
        description="""
            The Meta Llama 3.1 collection of multilingual large language
            models (LLMs) is a collection of pretrained and instruction tuned
            generative models in 8B, 70B and 405B sizes. The Llama 3.1
            instruction tuned text only models (8B, 70B, 405B) are optimized
            for multilingual dialogue use cases and outperform many of the
            available open source and closed chat models on common industry
            benchmarks. 405B model is the most capable from the Llama 3.1
            family. This model is served in FP8 closely matching reference
            implementation.""",
        has_tools=True,
        urls=["https://fireworks.ai/models/fireworks/llama-v3p1-405b-instruct"]
    )
    FIREWORKS_LLAMA_31_70B_INSTRUCT = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks llama 3.1 70B instruct",
        api_reference="accounts/fireworks/models/llama-v3p1-70b-instruct",
        description="""
            The Meta Llama 3.1 collection of multilingual large language
            models (LLMs) is a collection of pretrained and instruction tuned
            generative models in 8B, 70B and 405B sizes. The Llama 3.1
            instruction tuned text only models (8B, 70B, 405B) are optimized
            for multilingual dialogue use cases and outperform many of the
            available open source and closed chat models on common industry
            benchmarks.""",
        has_tools=True,
        urls=["https://fireworks.ai/models/fireworks/llama-v3p1-70b-instruct"]
    )
    FIREWORKS_LLAMA_33_70B_INSTRUCT = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks llama 3.3 70B instruct",
        api_reference="accounts/fireworks/models/llama-v3p3-70b-instruct",
        description="""
            Llama 3.3 70B Instruct is the December update of Llama 3.1 70B.
            The model improves upon Llama 3.1 70B (released July 2024) with
            advances in tool calling, multilingual text support, math and
            coding. The model achieves industry leading results in reasoning,
            math and instruction following and provides similar performance as
            3.1 405B but with significant speed and cost improvements.""",
        has_tools=True,
        urls=["https://fireworks.ai/models/fireworks/llama-v3p3-70b-instruct"]
    )
    FIREWORKS_MIXTRAL_MOE_8X22B_INSTRUCT = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks mixtral moe 8x22B instruct",
        api_reference="accounts/fireworks/models/mixtral-8x22b-instruct",
        description="""
            Mixtral MoE 8x22B Instruct v0.1 is the instruction-tuned version
            of Mixtral MoE 8x22B v0.1 and has the chat completions API
            enabled.""",
        has_tools=True,
        urls=["https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct"]
    )
    FIREWORKS_QWEN_QWQ_32B_PREVIEW = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks qwen qwq 32b preview",
        api_reference="accounts/fireworks/models/qwen-qwq-32b-preview",
        description="""
            Qwen QwQ model focuses on advancing AI reasoning, and showcases the
            power of open models to match closed frontier model performance.
            QwQ-32B-Preview is an experimental release, comparable to o1 and
            surpassing GPT-4o and Claude 3.5 Sonnet on analytical and reasoning
            abilities across GPQA, AIME, MATH-500 and LiveCodeBench benchmarks.
            Note: This model is served experimentally as a serverless model. If
            you're deploying in production, be aware that Fireworks may undeploy
            the model with short notice.""",
        has_tools=True,
        urls=["https://fireworks.ai/models/fireworks/qwen-qwq-32b-preview"]
    )
    FIREWORKS_QWEN_25_CODER_32B_INSTRUCT = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks qwen 2.5 coder 32b instruct",
        api_reference="accounts/fireworks/models/qwen2p5-coder-32b-instruct",
        description="""
        Qwen2.5-Coder is the latest series of Code-Specific Qwen large
        language models (formerly known as CodeQwen). Note: This model is
        served experimentally as a serverless model. If you're deploying in
        production, be aware that Fireworks may undeploy the model with short
        notice.
        """,
        has_tools=False,  # ?
        urls=["https://fireworks.ai/models/fireworks/qwen2p5-coder-32b-instruct"]
    )
    FIREWORKS_DEEPSEEK_R1 = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks deepseek r1",
        api_reference="accounts/fireworks/models/deepseek-r1",
        description="""
        DeepSeek-R1 is a state-of-the-art large language model optimized with
        reinforcement learning and cold-start data for exceptional reasoning,
        math, and code performance.
        """,
        has_tools=False,
        urls=["https://fireworks.ai/models/fireworks/deepseek-r1"]
    )
    FIREWORKS_ = LargeLanguageModelsAPIInfo(
        company="Fireworks",
        generic_name="fireworks deepseek v3",
        api_reference="accounts/fireworks/models/deepseek-v3",
        description="""
        A a strong Mixture-of-Experts (MoE) language model with 671B total
        parameters with 37B activated for each token from Deepseek.
        """,
        has_tools=False,
        urls=["https://fireworks.ai/models/fireworks/deepseek-v3"]
    )
    # FIREWORKS_ = LargeLanguageModelsAPIInfo(
    #     company="Fireworks",
    #     generic_name="",
    #     api_reference="",
    #     description="""
    #     """,
    #     has_tools=False,
    #     urls=[""]
    # )
    # TODO: additional fireworks llm models -> https://fireworks.ai/models


# ---- classes and functions ----
# Note: The above tests for invalid values (temperature, max_tokens, max_retries, timeout)
# will not raise ValueError by default because the dataclass does not enforce these constraints.
# You would need to add validation logic in the __post_init__ method of the dataclass to enforce these constraints.
@dataclass
class LargeLanguageModelConfig:
    """
    Configuration settings for initializing a large language model.

    This dataclass encapsulates the necessary parameters to initialize and configure
    various language models (OpenAI, Anthropic, Fireworks, Ollama) with consistent
    settings across different providers.

    Attributes:
        id (LargeLanguageModelsEnum): The identifier for the specific model to use,
            selected from the LargeLanguageModelsEnum options.
        api_key (str): The authentication key for the model's API service.
            Not required for local models like Ollama.
        temperature (float): Controls randomness in the model's output.
            Higher values (e.g., 0.8) make output more random,
            lower values (e.g., 0.2) make it more deterministic.
        max_tokens (int): The maximum number of tokens to generate in the response.
        max_retries (int): Number of retry attempts for failed API calls. Defaults to 3.
        timeout (Optional[float]): Maximum time in seconds to wait for a response.
            Defaults to None (no timeout).

    Example:
        config = LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O,
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000,
            timeout=30.0
        )
    """
    id: LargeLanguageModelsEnum
    api_key: str
    temperature: float
    max_tokens: int
    max_retries: int = field(default=3)
    timeout: Optional[float] = field(default=None)

    def __post_init__(self):
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be greater than 0.")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative.")
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("Timeout must be non-negative.")




# TODO: add a warning when using an expensive model!
def get_llm_model(model_config: LargeLanguageModelConfig):
    """
    Creates and returns a configured language model instance based on the provided configuration.

    This function serves as a factory method that instantiates the appropriate chat model
    (ChatAnthropic, ChatOpenAI, ChatFireworks, or ChatOllama) based on the model ID in the
    configuration. It applies all configuration parameters consistently across different
    model providers.

    Args:
        model_config (LargeLanguageModelConfig): Configuration object containing all necessary
            parameters to initialize the language model, including model ID, API key,
            temperature, token limits, and timeout settings.

    Returns:
        Union[ChatAnthropic, ChatOpenAI, ChatFireworks, ChatOllama]: An initialized chat model
        instance ready for use. If an unrecognized model ID is provided, defaults to
        OLLAMA_LLAMA_32_3B as a fallback.

    Notes:
        - Different providers may have slightly different parameter names (e.g., 'max_tokens'
            vs 'max_tokens_to_sample' vs 'num_predict')
        - Ollama models don't require an API key as they run locally
        - The function uses pattern matching to select the appropriate model class and
            configuration

    Example:
        config = LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O,
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )
        model = get_llm_model(config)
        # model is now ready for chat completions
    """
    match model_config.id.company:
        case "Ollama":
            return ChatOllama(
                model=model_config.id.api_reference,
                temperature=model_config.temperature,
                num_predict=model_config.max_tokens,
            )
        case "Anthropic":
            return ChatAnthropic(
                model_name=model_config.id.api_reference,
                api_key=model_config.api_key,
                temperature=model_config.temperature,
                max_tokens_to_sample=model_config.max_tokens,
                timeout=model_config.timeout,
                max_retries=model_config.max_retries,
                stop=None,
            )
        case "OpenAI":
            return ChatOpenAI(
                model=model_config.id.api_reference,
                api_key=model_config.api_key,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                max_retries=model_config.max_retries,
                # other params...
            )
        case "Fireworks":
            return ChatFireworks(
                model=model_config.id.api_reference,
                api_key=model_config.api_key,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                max_retries=model_config.max_retries,
                # other params...
            )
        case _:
            raise ValueError("Unknown model requested in get_llm_model().")
