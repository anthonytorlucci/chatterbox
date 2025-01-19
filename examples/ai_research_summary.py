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
        has_vector_dbs=True,
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
            "https://aiindex.stanford.edu/report/",
            "https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://www.ibm.com/think/topics/machine-learning",
        ],
        chunk_size=1024,
        chunk_overlap=256,
        k_results=3,
    )
    arxiv_context_retriever_node = ArxivContextRetriever(
        max_docs_to_load=2,
        chunk_size=1024,
        chunk_overlap=256,
        k_results=2,
    )
    vdbs_context_retriever_node = VdbsContextRetriever(
        db_dir=Path(__file__).parent.parent.joinpath("vdbs_documents", "chroma_research_notes_ollama_emb_db"),
        collection_names=[
            "medium_generative_driven_design",
            "medium_neural_networks_are_fundamentally_bayesian",
        ],
        k_results=3,
    )
    research_context_grader_node = ResearchContextGrader(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=2000
        ),
    )
    research_context_summarizer_node = ResearchContextSummarizer(
        model_config=LargeLanguageModelConfig(
            id=LargeLanguageModelsEnum.OLLAMA_MARCO_01_7B,
            api_key="",
            temperature=0.25,
            max_tokens=6000  # increase depending of relevant documents and size of each document
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

    # user_prompt = "Summarize current research on artificial intelligence."
    user_prompt = "What is the current research on machine learning and artificial intelligence."

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


# ---
# **Document 1 Summary:**
#
# This document discusses the evolution of benchmarking in generative models, highlighting a shift from computerized rankings to human evaluations such as the Chatbot Arena Leaderboard. The emphasis on public sentiment reflects its growing importance in assessing AI progress. Additionally, it addresses the environmental impact of AI systems, emphasizing the need for sustainable practices.
#
# **Document 2 Summary:**
#
# This chapter explores diversity trends within AI by analyzing data from the Computing Research Association (CRA) and Informatics Europe. It provides insights into the state of diversity in American and Canadian computer science departments. The document also touches on societal inequalities and biases present in AI, concluding with a discussion on the environmental footprint of AI systems.\n\n**Document 3 Summary:**\n\nThe chapter investigates diversity trends in AI by utilizing data from the Computing Research Association (CRA) and Informatics Europe to assess diversity within American and Canadian computer science departments. It discusses societal inequalities and biases in AI, and concludes with an exploration of the environmental impact of AI systems.\n\n**Document 4 Summary:**\n\nThis research paper, published in the Journal of Artificial Intelligence Research, surveys the integration of vision and language research. Authored by Aditya Mogadala, Marimuthu Kalimuthu, and Dietrich Klakow, it reviews tasks, datasets, and methods within this field. The document also addresses societal inequalities and biases in AI and concludes with a discussion on the environmental footprint of AI systems.\n\n**Document 5 Summary:**\n\nThe chapter examines diversity trends in AI using data from the Computing Research Association (CRA) and Informatics Europe to evaluate diversity in American and Canadian computer science departments. It highlights societal inequalities and biases within AI, concluding with an exploration of AI systems' environmental impact.\n\n**Document 6 Summary:**\n\nThis document delves into diversity trends in AI by analyzing data from the Computing Research Association (CRA) and Informatics Europe. It provides insights into diversity in American and Canadian computer science departments and discusses societal inequalities and biases in AI, concluding with an exploration of the environmental footprint of AI systems.\n\n**Document 7 Summary:**\n\nThe chapter explores diversity trends within AI through data from the Computing Research Association (CRA) and Informatics Europe, focusing on diversity in American and Canadian computer science departments. It addresses societal inequalities and biases in AI and concludes by examining the environmental impact of AI systems.
# ---
