"""
research_context_summarizer

Summarizes the available context retrieved.
"""

# standard lib
# from typing import (
#     Type,
#     List,
#     Optional
# )
import logging

# third party
#from pydantic import BaseModel, Field

# langchain
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    #MessagesPlaceholder
)
# from langchain_core.runnables import Runnable

# langgraph

# local
from chatterbox.language_models import (
    LargeLanguageModelConfig,
    get_llm_model
)
from chatterbox.nodes.triage import SYSTEM_PROMPT
from chatterbox.researcher_interface import (
    ResearcherInterface
)

SYSTEM_PROMPT = """
**Objective:** You are an intelligent summarization agent tasked with condensing the content of a set of documents retrieved from a database. Your goal is to provide clear, concise, and informative summaries that capture the main ideas and key points of each document.

**Instructions:**

Input Format:

You will receive a list of documents in text format. Each document may vary in length and complexity.
The documents may cover diverse topics and may include various types of content (e.g., research articles, reports, articles, etc.).

Summarization Guidelines:

* For each document, generate a summary that is no longer than 250 words.
* Focus on identifying and articulating the main arguments, findings, and conclusions of the document.
* Maintain the original meaning and context of the document while ensuring clarity and coherence in your summary.
* Avoid including personal opinions or interpretations; strictly summarize the content presented in the documents.

Output Format:

* Present your summaries in a numbered list format, corresponding to the order of the documents provided.
* Each summary should be labeled with the document number (e.g., "Document 1 Summary:").

Quality Assurance:

* Ensure that your summaries are free of grammatical errors and are easy to read.
* If a document is particularly complex, prioritize summarizing the most critical information over less important details.

# Examples

**Example 1:**

Input Document:
"Recent studies in marine biology have focused on the significant decline in coral reef health due to climate change. Researchers have identified increased water temperatures and acidification as primary causes affecting coral ecosystems. The study emphasizes the urgency in implementing conservation strategies to protect these vital marine habitats."

Document 1 Summary:
"Research has shown a decline in coral reef health primarily due to climate change. Key findings identify increased water temperatures and acidification as major threats. Urgent conservation strategies are emphasized to protect marine habitats."

**Example 2:**

Input Document:
"The report evaluates the economic impact of renewable energy investment in rural areas. It finds significant job creation and local economic growth as major benefits. The analysis also highlights reduced energy costs and increased energy security."

Document 2 Summary:
"Investing in renewable energy in rural regions leads to job creation and economic growth, with reduced energy costs and enhanced energy security as additional benefits."
"""

USER_PROMPT = """
Use the following research prompt:

\"""{research_prompt}\"""

to summarize the retrieved documents:

\"""{all_context}.\"""
"""

### subclass ResearcherInterface

class ResearchContextSummarizer(ResearcherInterface):
    """
    A class responsible for summarizing research context retrieved from various sources.

    This class processes and summarizes content from multiple document sources (PDF, web, arXiv, vector databases)
    using a large language model. It implements the ResearcherInterface and provides functionality
    to generate concise, informative summaries while maintaining the original context and meaning.

    Attributes:
        NAME (str): Identifier for the researcher agent, set to "research_context_summarizer"
        research_context_summarizer: A chain combining a prompt template and language model for summarization

    Args:
        model_config (LargeLanguageModelConfig): Configuration for the language model to be used for summarization

    Example:
        ```python
        model_config = LargeLanguageModelConfig(...)
        summarizer = ResearchContextSummarizer(model_config)
        result = summarizer(state_dict)
        ```

    The summarizer uses a structured prompt that:
    - Identifies and articulates main arguments, findings, and conclusions
    - Limits summaries to 250 words per document
    - Maintains original meaning while ensuring clarity
    - Presents summaries in a numbered list format
    - Avoids personal interpretations

    Raises:
        ValueError: If all context lists in the input state are empty
    """
    NAME = "research_context_summarizer"
    def __init__(self, model_config: LargeLanguageModelConfig):
        llm_summarizer = get_llm_model(model_config=model_config)

        # Prompt
        sys_prompt=PromptTemplate(
            input_variables=[],
            template=SYSTEM_PROMPT
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

        human_prompt: PromptTemplate = PromptTemplate(
            input_variables=["research_prompt", "all_context"],
            template=USER_PROMPT
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

        agent_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])


        self.research_context_summarizer = agent_prompt | llm_summarizer

    def __call__(self, state: dict):
        """
        Summarize the retrieved context.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict):
        """
        logging.info("---SUMMARIZE PROMPT---")
        all_context = state.get("pdf_context", [])
        all_context.extend(state.get("web_context", []))
        all_context.extend(state.get("arxiv_context", []))
        all_context.extend(state.get("vdbs_context", []))


        if research_prompt := state.get("research_prompt") and len(all_context) > 0:
            result = self.research_context_summarizer.invoke(
                input={
                    "research_prompt": research_prompt,
                    "all_context": "\n\n".join([doc.page_content for doc in all_context]),
                },
                #config=rconfig???
            )
            return {
                "calling_agent": self.NAME,
                "summarized_context": result.content,
                "messages": state["messages"] + [result]
            }
        else:
            #--raise ValueError("All context lists are empty or research prompt is empty.")
            message = AIMessage(content="All context lists are empty or research prompt is empty. Try increasing the number of main ideas or adding additional sources for research.")
            return {
                "calling_agent": self.NAME,
                "summarized_context": message.content,
                "messages": state["messages"] + [message]
            }
