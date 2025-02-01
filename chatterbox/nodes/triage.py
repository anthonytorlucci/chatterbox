"""
triage
"""

# standard lib
import logging
from typing import (
    List,
)

# third party
from pydantic import BaseModel, Field

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# from langchain_core.runnables import Runnable

# langgraph

# local
from chatterbox.language_models import (
    LargeLanguageModelConfig,
    get_llm_model
)
from chatterbox.researcher_interface import (
    ResearcherInterface
)

SYSTEM_PROMPT = """
Your primary objective is to serve as a triage agent in our multi-agent
research system. Your role is crucial in accurately extracting main ideas from
the input text and initializing the graph state for subsequent agents. This
task involves understanding the core concepts and key points of the given
research prompt or question.

To initiate this process, please format your response using a structured list
format. Each item in the list should represent a significant idea or concept
identified from the input. Use lower case characters unless the word is a
proper noun.

Remember, precision and comprehensiveness are vital at this stage as your
extraction will guide all subsequent agents' actions and information retrieval
processes. Your understanding should go beyond mere keywords; it should
encapsulate the essence or core themes of the input text.

To achieve this, you may employ techniques such as semantic analysis or topic
modeling to identify the most significant elements within the provided
research prompt or question. However, your primary tool for this task is a
large language model which you will use to analyze and interpret the input
text.

Finally, after extracting these main ideas, prepare to initialize the graph
state with these key concepts. This initialization sets the foundation for all
subsequent processing by other agents in our system.

Please begin by defining your extraction objective clearly before proceeding
with the analysis.

# Examples

1. **Input Prompt**: "What is machine learning?"
- **Extracted Main Ideas**:
- artificial intelligence
- machine learning
- statistics
- pattern recognition

2. **Input Prompt**: "Explain the impact of climate change on global agriculture."
- **Extracted Main Ideas**:
- climate change
- global agriculture
- environmental impact
- food security

3. **Input Prompt**: "Discuss the applications of blockchain technology in finance."
- **Extracted Main Ideas**:
- blockchain technology
- finance
- cryptocurrency
- transaction security

4. **Input Prompt**: "What are some wavelet methods for time series analysis and signal processing?"
- **Extracted Main Ideas**:
- wavelet transform
- multiresolution analysis
- Daubechies wavelets
- signal processing
- fast Fourier transform
- time-frequency localization

# Output Format

- Structured list format with each item representing a key concept or main idea from the input text.

# Notes

- Ensure the identification goes beyond keywords to capture the core themes.
- Use semantic analysis or topic modeling if necessary, but prioritize using the large language model for analysis.
- Examples should guide your understanding and approach, applying similar logic and extraction methods to new inputs.
"""

USER_PROMPT = """
Generate a list of {num_main_ideas} main ideas and key concepts from the prompt:

{research_prompt}.
"""

class MainIdeas(BaseModel):
    """Main ideas, keywords, or key concepts from the prompt."""

    main_ideas: List[str] = Field(
        description="main ideas, keywords, and key phrases extracted from the input prompt."
    )

### subclass ResearcherInterface

class Triage(ResearcherInterface):
    """A specialized agent that extracts main ideas and key concepts from
    research prompt and initializes the graph state.

    This class is a component of a multi-agent research system built with LangGraph.
    It implements the ResearcherInterface and uses a large language model to analyze
    input prompts and identify their core concepts and main ideas.

    The agent processes the input through a chain of:
    1. A system prompt that defines the extraction objective
    2. A human prompt template that formats the input
    3. An LLM that performs the extraction
    4. A tool to convert the LLM's response into a structured list

    Attributes:
        NAME (str): The identifier for this agent type ("main_ideas_extractor")
        main_ideas_extractor: A chain combining the prompt template and LLM with tools

    Args:
        model_config (LargeLanguageModelConfig): Configuration for the language model
            to be used for extraction
        max_num_main_ideas (int): maximum number of output main ideas, default is 5

    Additional Notes:

    This agent is the first agent in the workflow. It behaves like a "triage"
    agent which means it needs to prioritize and assess situations quickly.

    In a multi-agent system, agents work together to achieve a goal, and each
    has a specific role. The triage agent, being first, needs to set the tone
    for the entire process.

    This agent receives input text in the form a research question or a
    statement that requires additional research and its job is to parse that
    input to find the crucial points.

    When solving a complex problem, this agent first needs to understand what
    the core issues are before delving into details.

    For example, if it's processing a user's question, it needs to identify the
    key entities, keywords, or phrases that are essential to understanding what
    the user is asking for. These main ideas will then be used by other agents
    to retrieve relevant information, perform actions, or generate responses.

    It has two primary goals: extract main ideas and initialize the state.

    Extracting main ideas is important because all the additional context
    retrieved later depends on these key words or phrases. So, if it misses
    something here, everything that follows could be off track because
    subsequent agents are relying on this initial assessment. Given that all
    additional context retrieved depends on these key words or phrases, it's
    crucial that this agent does its job accurately. It's similar to keyword
    extraction in information retrieval systems, where the system identifies
    the most relevant terms to search for in a database. But here, it's more
    nuanced because it's not just about keywords; it's about extracting main
    ideas, which could involve some level of understanding or interpretation.
    This agent needs to have some cognitive abilities, like using a large
    language model, to comprehend the input and distill it down to its core
    components. It might use techniques like semantic analysis or topic
    modeling to identify the most significant elements.

    It also sets up the initial state (parameters) that other agents will use
    as they proceed with their tasks. So, it's like establishing the starting
    point for the entire workflow.

    Perhaps there should be some mechanism for quality assurance or
    double-checking the triage agent's output, especially in critical systems
    where mistakes could have significant consequences.

    In summary, this triage agent is a crucial component in the multi-agent
    system, responsible for quickly and accurately extracting main ideas from
    input and initializing the state, which sets the foundation for all
    subsequent processing.

    Example:
        ```python
        config = LargeLanguageModelConfig(...)
        extractor = MainIdeasExtractor(config)
        state = {"research_prompt": "Explain the impact of climate change on biodiversity"}
        result = extractor(state)
        # Returns dict with main_ideas list and calling_agent identifier
        ```
    """
    NAME = "triage"
    def __init__(
        self,
        model_config: LargeLanguageModelConfig,
        max_num_main_ideas: int = 5
    ):
        """
        Initialize a Triage agent that extracts main ideas from research prompts.

        This constructor sets up the agent's language model and prompt chain for
        extracting key concepts. It configures a system prompt that defines the
        extraction objectives, a human prompt template for formatting input, and
        combines these with an LLM to create the main ideas extraction pipeline.

        Args:
            model_config (LargeLanguageModelConfig): Configuration object for the
                large language model to be used for extraction. This includes
                settings like model name, temperature, and other parameters.
            max_num_main_ideas (int, optional): Maximum number of main ideas to
                extract from the input prompt. Defaults to 5.

        Attributes:
            _max_num_main_ideas (int): Stores the maximum number of main ideas to extract
            main_ideas_extractor: A chain combining prompt templates and LLM for
                extracting main ideas with structured output

        Example:
        ```python
        config = LargeLanguageModelConfig(model_name="gpt-4")
        triage_agent = Triage(model_config=config, max_num_main_ideas=3)
        ```

        Note:
        The agent uses a structured prompt chain that includes:
            1. A system prompt defining the extraction objective
            2. A human prompt template for formatting the input
             3. An LLM configured to output structured data (MainIdeas model)
        """
        self._max_num_main_ideas = max_num_main_ideas
        llm_extractor = get_llm_model(model_config=model_config).with_structured_output(MainIdeas)

        # Prompt
        sys_prompt = PromptTemplate(
            input_variables=[],
            template=SYSTEM_PROMPT
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

        human_prompt: PromptTemplate = PromptTemplate(
            input_variables=["research_prompt", "num_main_ideas"],
            template=USER_PROMPT
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

        agent_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])


        self.main_ideas_extractor = agent_prompt | llm_extractor

    def __call__(self, state: dict):
        """
        Process the input state to extract main ideas and initialize the research graph state.

        This method serves as the primary execution point for the Triage agent. It
        analyzes the latest message in the state, extracts key concepts, and
        prepares the initial state for the research workflow. The method ensures
        the research process begins with a clear understanding of the core concepts
        that need to be investigated.

        Args:
            state (dict): The current graph state containing:
                - messages: List of chat messages, with the last message being the
                  research prompt to analyze
                - Other optional state parameters

        Returns:
            dict: An updated state dictionary containing:
                - calling_agent (str): Identifier for this agent ("triage")
                - main_ideas (List[str]): Extracted key concepts and main ideas
                - research_prompt (str): The original research prompt
                - messages (List): Updated message history including the extraction results
                - recursion_limit (int): Maximum number of recursive operations allowed

        Raises:
            ValueError: If no messages are found in the input state

        Example:
        ```python
        initial_state = {
            "messages": [HumanMessage(content="Explain quantum computing's impact on cryptography")]
        }
        updated_state = triage_agent(initial_state)
        # Returns state with extracted main ideas and initialized parameters
        ```

        Note:
            - The method processes only the last message in the state's message history
            - The number of extracted main ideas is limited by _max_num_main_ideas
            - Adds an AIMessage to the message history summarizing the extracted ideas
            - Sets up initial parameters needed for the research workflow
        """
        logging.info("---EXTRACT MAIN IDEAS AND INITIALIZE STATE---")
        if messages := state.get("messages"):
            research_prompt = messages[-1].content
            ideas = self.main_ideas_extractor.invoke(
                input={
                    "research_prompt": research_prompt,
                    "num_main_ideas": self._max_num_main_ideas
                },
                #config=rconfig???
            )
            main_ideas = ideas.main_ideas[:self._max_num_main_ideas]
            ai_message = "Extracted main ideas: " + ", ".join(main_ideas)
            return {
                "calling_agent": self.NAME,
                "main_ideas": main_ideas,
                "research_prompt": research_prompt,
                "messages": state["messages"] + [AIMessage(content=ai_message)],
                "requires_pdf_context": True,  # default state is True; manages by supervisor and context grader
                "requires_vdbs_context": True,  # default state is True; manages by supervisor and context grader
                "requires_web_context": True,  # default state is True; manages by supervisor and context grader
                "requires_arxiv_context": True,  # default state is True; manages by supervisor and context grader
                "recursion_limit": 30
            }
        else:
            raise ValueError("No messages found.")
