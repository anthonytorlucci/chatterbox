"""
research_context_supervisor

Supervisor to retrieve context from various sources including:
    - pdfs
    - web urls
    - arxiv
    - pre-built vector databases
"""

# standard lib
import logging

# third party

# langchain
from langchain_core.messages.ai import AIMessage

# langgraph

# local
from chatterbox.researcher_interface import (
    ResearcherInterface
)

### subclass ResearcherInterface

class ResearchContextSupervisor(ResearcherInterface):
    """
    A supervisor agent that manages the retrieval of research context from multiple sources.

    This class acts as a routing node in a LangGraph multi-agent system, determining which
    context retrieval agent should be called next based on the current state and available
    context sources.

    The supervisor checks for the presence of context from different sources in the following order:
    1. PDF documents (if PDF paths are provided)
    2. Vector databases (if DB directories are provided)
    3. Web documents
    4. ArXiv papers

    Attributes:
        NAME (str): The identifier for this agent node
        _has_pdf_paths (bool): Indicates if PDF document paths are available for context
        _has_vector_dbs (bool): Indicates if vector database directories are provided and should be retrieved

    Example:
        supervisor = ResearchContextSupervisor(
            has_pdf_paths=False,
            has_vector_dbs=False
        )
        next_step = supervisor(current_state)
    """
    NAME = "research_context_supervisor"
    def __init__(self,
        has_pdf_paths: bool = False,
        has_vector_dbs: bool = False
    ):
        """
        Initialize the ResearchContextSupervisor.

        Args:
            has_pdf_paths (bool, optional): Flag indicating whether PDF document paths are
                available for context retrieval. Defaults to False.
            has_vector_dbs (bool, optional): Flag indicating whether vector database
                directories are available for context retrieval. Defaults to False.
        """
        self._has_pdf_paths = has_pdf_paths
        self._has_vector_dbs = has_vector_dbs

    def __call__(self, state: dict):
        """
        Supervises the generation of relevant context to the user prpompt.

        Args:
            state (dict): The current graph state

        Returns:
            (str): the next node to route.
        """
        logging.info("---SUERPVISE CONTEXT GENERATION---")
        pdf_context = state.get("pdf_context", [])
        web_context = state.get("web_context", [])
        arxiv_context = state.get("arxiv_context", [])
        vdbs_context = state.get("vdbs_context", [])

        has_pdf_context = True if len(pdf_context) > 0 else False
        has_web_context = True if len(web_context) > 0 else False
        has_arxiv_context = True if len(arxiv_context) > 0 else False
        has_vdbs_context = True if len(vdbs_context) > 0 else False

        requires_pdf_context = all([self._has_pdf_paths, not has_pdf_context])
        requires_vdbs_context = all([self._has_vector_dbs, not has_vdbs_context])
        requires_web_context = all([not has_web_context])
        requires_arxiv_context = all([not has_arxiv_context])

        pdf_message = "pdf required, " if requires_pdf_context else ""
        vdbs_message = "vdbs required, " if requires_vdbs_context else ""
        web_message = "web required, " if requires_web_context else ""
        arxiv_message = "arxiv required" if requires_arxiv_context else ""
        ai_message = "".join(["Context generation supervisor should fetch: ", pdf_message, vdbs_message, web_message, arxiv_message])

        return {
            "calling_agent": self.NAME,
            "requires_pdf_context": requires_pdf_context,
            "requires_vdbs_context": requires_vdbs_context,
            "requires_web_context": requires_web_context,
            "requires_arxiv_context": requires_arxiv_context,
            "messages": state["messages"] + [AIMessage(content=ai_message)]
        }
