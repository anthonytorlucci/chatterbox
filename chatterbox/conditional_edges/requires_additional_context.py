# standard lib
import logging
# third party
# langchain
# langgraph
# local


def route_to_context_retriever_or_summarize(state):
    """
    Determines whether to route to a context retriever or summarize the research context.

    Args:
        state (dict): The current graph state

    Returns:
        str: next node to call
    """
    if state.get("requires_pdf_context", False):
        logging.info("---DECISION: REQUIRES PDF CONTEXT---")
        return "pdf" # -> PdfContextRetriever
    elif state.get("requires_vdbs_context", False):
        logging.info("---DECISION: REQUIRES VDBS CONTEXT---")
        return "vdbs" # -> VdbsContextRetriever
    elif state.get("requires_web_context", False):
        logging.info("---DECISION: REQUIRES WEB CONTEXT---")
        return "web" # -> WebContextRetriever
    elif state.get("requires_arxiv_context", False):
        logging.info("---DECISION: REQUIRES ARXIV CONTEXT---")
        return "arxiv" # -> ArxivContextRetriever
    else:
        logging.info("---DECISION: PROMOTE TO SUMMARIZE PROMPT---")
        return "summarize" # -> ResearchContextSummarizer
