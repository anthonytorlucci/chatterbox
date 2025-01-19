# standard library
import os
import sys
import logging
from pathlib import Path
# third party
# langchain
from langchain_core.messages import HumanMessage, AIMessage
# langgraph
# local
sys.path.append(str(Path(__file__).parent.parent))
from chatterbox.language_models import LargeLanguageModelConfig, LargeLanguageModelsEnum
from chatterbox.nodes.triage import Triage
from chatterbox.nodes.research_context_supervisor import ResearchContextSupervisor
from chatterbox.nodes.pdf_context_retriever import PdfContextRetriever
from chatterbox.nodes.web_context_retriever import WebContextRetriever
from chatterbox.nodes.arxiv_context_retriever import ArxivContextRetriever
from chatterbox.nodes.vdbs_context_retriever import VdbsContextRetriever
from chatterbox.nodes.research_context_grader import ResearchContextGrader
from chatterbox.nodes.research_context_summarizer import ResearchContextSummarizer
from chatterbox.workflows import (
    build_context_summary_graph
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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found.")
# TODO: LANGSMITH_API_KEY
USER_AGENT = os.getenv("USER_AGENT")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s", # Customize the log message format
)

if __name__ == "__main__":
    triage_node = Triage(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=1000
        ),
        max_num_main_ideas=2,
    )
    research_context_supervisor_node = ResearchContextSupervisor(
        has_pdf_paths=False,
        has_vector_dbs=False,
        has_urls=True,
        use_arxiv_search=False,
    )
    pdf_context_retriever_node = PdfContextRetriever(
        pdf_paths=[],
        chunk_size=400,
        chunk_overlap=100,
        k_results=3,
    )
    web_context_retriever_node = WebContextRetriever(
        urls=[
            "https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications",
            #"https://www.xenonstack.com/blog/graph-neural-network-applications",
            #"https://www.assemblyai.com/blog/ai-trends-graph-neural-networks/",
            "https://en.wikipedia.org/wiki/Graph_neural_network",
            #"https://www.sciencedirect.com/science/article/pii/S2211675324000137",
            #"https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-024-03199-x",
            #"https://onlinelibrary.wiley.com/doi/full/10.1002/int.23087",
            #"https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial",
            "https://distill.pub/2021/gnn-intro/",
        ],
        chunk_size=800,
        chunk_overlap=200,
        k_results=2,
    )
    arxiv_context_retriever_node = ArxivContextRetriever(
        max_docs_to_load=3,
        chunk_size=400,
        chunk_overlap=100,
        k_results=3,
    )
    vdbs_context_retriever_node = VdbsContextRetriever(
        vdbs=[],
        k_results=3,
        fetch_k_results=5,
    )
    research_context_grader_node = ResearchContextGrader(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
            api_key=str(OPENAI_API_KEY),
            temperature=0.0,
            max_tokens=2000
        ),
    )
    research_context_summarizer_node = ResearchContextSummarizer(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OLLAMA_PHI4_14B,
            api_key="",
            temperature=0.25,
            max_tokens=4000  # increase depending of relevant documents and size of each document
        ),
    )


    workflow = build_context_summary_graph(
        triage_node=triage_node,
        research_context_supervisor_node=research_context_supervisor_node,
        pdf_context_retriever_node=pdf_context_retriever_node,
        web_context_retriever_node=web_context_retriever_node,
        arxiv_context_retriever_node=arxiv_context_retriever_node,
        vdbs_context_retriever_node=vdbs_context_retriever_node,
        research_context_grader_node=research_context_grader_node,
        research_context_summarizer_node=research_context_summarizer_node,
    )
    graph = workflow.compile()

    user_prompt = "What are graph neural networks in the context of deep learning?"

    # run graph
    try:
        message = AIMessage(content="")  # initialize message
        for output in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=user_prompt
                    ),
                ],
            }
        ):
            # print(output)
            for key, value in output.items():
                logging.info(f"Agent: output from node '{key}':")
                messages = value.get("messages", [])
                if len(messages) > 0:
                    message = messages[-1]
                    logging.info(message)

    except Exception as e:
        logging.error(e)
