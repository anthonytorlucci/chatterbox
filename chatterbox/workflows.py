"""Functions and classes to generate various supported workflows."""

# standard lib
from typing import (
    List,
    Sequence,
    Annotated,
    TypedDict
)

# third party

# langchain
from langchain_core.documents import (
    Document,
)
from langchain_core.messages import (
    BaseMessage,
)

# langgraph
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START

# local
from chatterbox.nodes.simple_chat import SimpleChat
from chatterbox.nodes.triage import Triage
from chatterbox.nodes.research_context_supervisor import ResearchContextSupervisor
from chatterbox.nodes.pdf_context_retriever import PdfContextRetriever
from chatterbox.nodes.web_context_retriever import WebContextRetriever
from chatterbox.nodes.arxiv_context_retriever import ArxivContextRetriever
from chatterbox.nodes.vdbs_context_retriever import VdbsContextRetriever
from chatterbox.nodes.research_context_grader import ResearchContextGrader
from chatterbox.nodes.research_context_summarizer import ResearchContextSummarizer
from chatterbox.conditional_edges.requires_additional_context import route_to_context_retriever_or_summarize

class SimpleChatAgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_simple_chat_graph(
    simple_chat_node: SimpleChat
):
    # Define a new graph
    workflow = StateGraph(SimpleChatAgentState)

    # Define the nodes we will cycle between
    workflow.add_node(simple_chat_node.NAME, simple_chat_node)
    workflow.add_edge(
        START,
        simple_chat_node.NAME
    )
    workflow.add_edge(
        simple_chat_node.NAME,
        END
    )
    return workflow

class ResearchAgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    github_context: List[Document]  # <- context from user_input_github_urls + any GitHub urls obtained in a web search
    web_context: List[Document]  # <- context from web context retriever
    pdf_context: List[Document]  # <- context obtained from pdf context retriever
    arxiv_context: List[Document]  # <- context obtained from the arxiv context retriever
    vdbs_context: List[Document]
    main_ideas: List[str]
    research_prompt: str
    calling_agent: str  # <- agent that indicated the call
    research_sources: List[str]  # <- source from each doc.metadata used in research plan and execute
    requires_pdf_context: bool
    requires_web_context: bool
    requires_arxiv_context: bool
    requires_vdbs_context: bool
    summarized_context: str
    research_plan: List[str]
    research_plan_past_steps: List[str]   # Annotated[List[Tuple], operator.add]
    recursion_limit: int

def build_context_summary_graph(
    triage_node: Triage,
    research_context_supervisor_node: ResearchContextSupervisor,
    pdf_context_retriever_node: PdfContextRetriever,
    web_context_retriever_node: WebContextRetriever,
    arxiv_context_retriever_node: ArxivContextRetriever,
    vdbs_context_retriever_node: VdbsContextRetriever,
    research_context_grader_node: ResearchContextGrader,
    research_context_summarizer_node: ResearchContextSummarizer,
):
    # Define a new graph
    workflow = StateGraph(ResearchAgentState)

    # Define the nodes we will cycle between
    workflow.add_node(triage_node.NAME, triage_node)
    workflow.add_node(research_context_supervisor_node.NAME, research_context_supervisor_node)
    workflow.add_node(pdf_context_retriever_node.NAME, pdf_context_retriever_node)
    workflow.add_node(web_context_retriever_node.NAME, web_context_retriever_node)
    workflow.add_node(arxiv_context_retriever_node.NAME, arxiv_context_retriever_node)
    workflow.add_node(vdbs_context_retriever_node.NAME, vdbs_context_retriever_node)
    workflow.add_node(research_context_grader_node.NAME, research_context_grader_node)
    workflow.add_node(research_context_summarizer_node.NAME, research_context_summarizer_node)

    workflow.add_edge(
        START,
        triage_node.NAME
    )
    workflow.add_edge(
        triage_node.NAME,
        research_context_supervisor_node.NAME
    )
    workflow.add_conditional_edges(
        research_context_supervisor_node.NAME,
        route_to_context_retriever_or_summarize,
        {
            "pdf": pdf_context_retriever_node.NAME,
            "vdbs": vdbs_context_retriever_node.NAME,
            "web": web_context_retriever_node.NAME,
            "arxiv": arxiv_context_retriever_node.NAME,
            "summarize": research_context_summarizer_node.NAME
        }
    )
    workflow.add_edge(
        pdf_context_retriever_node.NAME,
        research_context_grader_node.NAME,
    )
    workflow.add_edge(
        web_context_retriever_node.NAME,
        research_context_grader_node.NAME,
    )
    workflow.add_edge(
        arxiv_context_retriever_node.NAME,
        research_context_grader_node.NAME,
    )
    workflow.add_edge(
        vdbs_context_retriever_node.NAME,
        research_context_grader_node.NAME,
    )
    workflow.add_edge(
        research_context_grader_node.NAME,
        research_context_supervisor_node.NAME
    )
    workflow.add_edge(
        research_context_summarizer_node.NAME,
        END
    )

    return workflow
