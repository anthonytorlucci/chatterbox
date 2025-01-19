"""
Frontend streamlit app that provides a summary of the research context given a user prompt.
"""

# standard lib
import os
import re
import html
from pathlib import Path
from typing import (
    List,
    Tuple,
)
from enum import Enum
import logging
from datetime import datetime
from uuid import uuid4

# third party
from langchain_chroma.vectorstores import DEFAULT_K
import streamlit as st
from pydantic import (
    AnyUrl
)

# langchain
from langchain_core.messages import HumanMessage, AIMessage
#--from langchain_core.runnables.config import RunnableConfig

# langgraph
from langgraph.checkpoint.memory import MemorySaver

# local
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
from chatterbox.utils import is_valid_url

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

# ---- global variables ----
#--- Get the current date and time
#---now = datetime.now()
# Format the date and time as a string
current_time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s", # Customize the log message format
    # filename=Path(__file__).parent.joinpath("logs", f"{current_time_string}.log"),  # Log to a file
    # filemode="w" # Overwrite log file on each run. Use 'a' to append.
)

TRIAGE_LLMS = [
    LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
    LargeLanguageModelsEnum.OPENAI_GPT_4O,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_3_HAIKU,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_35_HAIKU,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_35_SONNET
]

GRADER_LLMS = [
    LargeLanguageModelsEnum.OPENAI_GPT_4O_MINI,
    LargeLanguageModelsEnum.OPENAI_GPT_4O,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_3_HAIKU,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_35_HAIKU,
    LargeLanguageModelsEnum.ANTHROPIC_CLAUDE_35_SONNET
]

SUMMARIZER_LLMS = [llm for llm in LargeLanguageModelsEnum]

# personal research notes not in GitHub
VDBS_DB_DIR = Path(__file__).parent.joinpath("vdbs_documents", "chroma_research_notes_ollama_emb_db")
print(VDBS_DB_DIR)
COLLECTION_NAMES = [
    "medium_generative_driven_design",
    "medium_neural_networks_are_fundamentally_bayesian"
]

# initial values in the user interface
INITIAL_NUM_MAIN_IDEAS = int(3)
INITIAL_CHUNK_SIZE = int(1024)  # 2**10
CHUNK_SIZE_MIN_VALUE = int(32)  # 2**5
CHUNK_SIZE_MAX_VALUE = int(32_768)  # 2**15
CHUNK_SIZE_STEP = int(1)
INITIAL_CHUNK_OVERLAP = int(256)  # 2**8
CHUNK_OVERLAP_MIN_VALUE = int(32)  # 2**5
CHUNK_OVERLAP_MAX_VALUE = int(32_768)  # 2**15
CHUNK_OVERLAP_STEP = int(1)
INITIAL_K_RESULTS = int(3)
K_RESULTS_MIN_VALUE = int(1)
K_RESULTS_MAX_VALUE = int(12)  # arbitrary max value
K_RESULTS_STEP = int(1)
INITIAL_USE_ARXIV_SEARCH = True
INITIAL_ARXIV_MAX_DOCS_TO_LOAD = 5
INITIAL_LLM_TEMPERATURE = 0.5
LLM_TEMPERATURE_MIN_VALUE = float(0.0)
LLM_TEMPERATURE_MAX_VALUE = float(1.0)
LLM_TEMPERATURE_STEP = float(0.01)
INITIAL_LLM_MAX_TOKENS = 1024
LLM_MAX_TOKENS_MIN_VALUE = int(32)
LLM_MAX_TOKENS_MAX_VALUE = int(128_000)
LLM_MAX_TOKENS_STEP = int(1)


# ---- streamlit app configuration and state
# streamlit page configuration
st.title("Chatterbox")

# triage
if "triage_node" not in st.session_state:
    st.session_state["triage_node"] = None
# supervisor
if "research_context_supervisor_node" not in st.session_state:
    st.session_state["research_context_supervisor_node"] = None
# pdf
if "pdf_context_retreiver_node" not in st.session_state:
    st.session_state["pdf_context_retreiver_node"] = None
# web
if "web_context_retreiver_node" not in st.session_state:
    st.session_state["web_context_retreiver_node"] = None
# arxiv
if "arxiv_context_retreiver_node" not in st.session_state:
    st.session_state["arxiv_context_retreiver_node"] = None
# vdbs
if "vdbs_context_retreiver_node" not in st.session_state:
    st.session_state["vdbs_context_retreiver_node"] = None
# grader
if "research_context_grader_node" not in st.session_state:
    st.session_state["research_context_grader_node"] = None
# summarizer
if "research_context_summarizer_node" not in st.session_state:
    st.session_state["research_context_summarizer_node"] = None
if "workflow" not in st.session_state:
    st.session_state["workflow"] = None
# if "runnable_config" not in st.session_state:
#     st.session_state["runnable_config"] = RunnableConfig(
#         configurable={
#             "thread_id": "0", #str(uuid4()),
#         }
#     )
if "should_update_workflow" not in st.session_state:
    st.session_state["should_update_workflow"] = False

def llm_api_key(company:str) -> str:
    match company:
        case "OpenAI":
            llm_api_key = OPENAI_API_KEY
        case "Anthropic":
            llm_api_key = ANTHROPIC_API_KEY
        case "Fireworks":
            llm_api_key = FIREWORKS_API_KEY
        case _:
            llm_api_key = ""
    return str(llm_api_key)

def get_valid_urls():
    url_inputs = [st.session_state["url_input_{}".format(i)] for i in range(10)]
    return [
        u for u in url_inputs if is_valid_url(u)
    ]

# ---- callbacks ----
def set_should_update_workflow_flag():
    st.session_state["should_update_workflow"] = True

def update_chat_workflow():
    """Callback to update the nodes and workflow when settings change"""
    logging.info(f""".... update_chat_workflow called.\n""")

    triage_llm_enum = [mdl for mdl in TRIAGE_LLMS if mdl.generic_name == st.session_state["triage_llm_name"]][0]  # TODO better heuristic
    logging.info(f""".... language model: {triage_llm_enum}""")
    triage_llm_api_key = llm_api_key(triage_llm_enum.company)

    research_context_grader_llm_enum = [mdl for mdl in GRADER_LLMS if mdl.generic_name == st.session_state["research_context_grader_llm_name"]][0]  # TODO better heuristic
    logging.info(f""".... language model: {research_context_grader_llm_enum}""")
    research_context_grader_llm_api_key = llm_api_key(research_context_grader_llm_enum.company)

    research_context_summarizer_llm_enum = [mdl for mdl in LargeLanguageModelsEnum if mdl.generic_name == st.session_state["research_context_summarizer_llm_name"]][0]  # TODO better heuristic
    logging.info(f""".... language model: {research_context_summarizer_llm_enum}""")
    research_context_summarizer_llm_api_key = llm_api_key(research_context_summarizer_llm_enum.company)

    # print("\n\n", "valid urls:\n", get_valid_urls(), "\n\n")
    valid_urls = get_valid_urls()
    #---print(f"\n\n !!! Runnable config: {st.session_state["runnable_config"]} !!! \n\n")
    st.session_state["triage_node"] = Triage(
        model_config=LargeLanguageModelConfig(
            id=triage_llm_enum,
            api_key=triage_llm_api_key,
            temperature=st.session_state["triage_temperature"],
            max_tokens=st.session_state["triage_max_tokens"]
        ),
        max_num_main_ideas=st.session_state["num_main_ideas"],
    )
    st.session_state["research_context_supervisor_node"] = ResearchContextSupervisor(
        has_pdf_paths=True if st.session_state["pdf_paths"] else False,
        has_vector_dbs=True if st.session_state["vdbs_collection_names"] else False,
        has_urls=True if valid_urls else False,
        use_arxiv_search=st.session_state["arxiv_search"],
    )
    st.session_state["pdf_context_retriever_node"] = PdfContextRetriever(
        pdf_paths=st.session_state["pdf_paths"],
        chunk_size=st.session_state["pdf_chunk_size"],
        chunk_overlap=st.session_state["pdf_chunk_overlap"],
        k_results=st.session_state["pdf_k_results"],
    )
    st.session_state["web_context_retriever_node"] = WebContextRetriever(
        urls=valid_urls,
        chunk_size=st.session_state["url_chunk_size"],
        chunk_overlap=st.session_state["url_chunk_overlap"],
        k_results=st.session_state["url_k_results"],
    )
    st.session_state["arxiv_context_retriever_node"] = ArxivContextRetriever(
        max_docs_to_load=st.session_state["arxiv_max_docs_to_load"],
        chunk_size=st.session_state["arxiv_chunk_size"],
        chunk_overlap=st.session_state["arxiv_chunk_overlap"],
        k_results=st.session_state["arxiv_k_results"],
    )
    st.session_state["vdbs_context_retriever_node"] = VdbsContextRetriever(
        db_dir=VDBS_DB_DIR,
        collection_names=st.session_state["vdbs_collection_names"],
        k_results=st.session_state["vdbs_k_results"],
    )
    st.session_state["research_context_grader_node"] = ResearchContextGrader(
        model_config=LargeLanguageModelConfig(
            id=research_context_grader_llm_enum,
            api_key=research_context_grader_llm_api_key,
            temperature=st.session_state["research_context_grader_temperature"],
            max_tokens=st.session_state["research_context_grader_max_tokens"]
        ),
    )
    st.session_state["research_context_summarizer_node"] = ResearchContextSummarizer(
        model_config=LargeLanguageModelConfig(
            id=research_context_summarizer_llm_enum,
            api_key=research_context_summarizer_llm_api_key,
            temperature=st.session_state["research_context_summarizer_temperature"],
            max_tokens=st.session_state["research_context_summarizer_max_tokens"]
        ),
    )
    st.session_state["workflow"] = build_context_summary_graph(
        triage_node=st.session_state["triage_node"],
        research_context_supervisor_node=st.session_state["research_context_supervisor_node"],
        pdf_context_retriever_node=st.session_state["pdf_context_retriever_node"],
        web_context_retriever_node=st.session_state["web_context_retriever_node"],
        arxiv_context_retriever_node=st.session_state["arxiv_context_retriever_node"],
        vdbs_context_retriever_node=st.session_state["vdbs_context_retriever_node"],
        research_context_grader_node=st.session_state["research_context_grader_node"],
        research_context_summarizer_node=st.session_state["research_context_summarizer_node"],
    )
    st.session_state["should_update_workflow"] = False



# ---- wigdet containers in sidebar ----
with st.sidebar:
    with st.container(border=True):
        st.header("triage")
        st.selectbox(
            label="triage model",
            options=[mdl.generic_name for mdl in TRIAGE_LLMS],
            index=0,
            key="triage_llm_name",
            on_change=set_should_update_workflow_flag
        )
        triage_temperature = st.number_input(
            label="temperature",
            min_value=LLM_TEMPERATURE_MIN_VALUE,
            max_value=LLM_TEMPERATURE_MAX_VALUE,
            value=INITIAL_LLM_TEMPERATURE,
            step=LLM_TEMPERATURE_STEP,
            key="triage_temperature",
            on_change=set_should_update_workflow_flag
        )
        triage_max_tokens = st.number_input(
            label="maximum number of tokens",
            min_value=LLM_MAX_TOKENS_MIN_VALUE,
            max_value=LLM_MAX_TOKENS_MAX_VALUE,
            value=INITIAL_LLM_MAX_TOKENS,
            step=LLM_MAX_TOKENS_STEP,
            key="triage_max_tokens",
            on_change=set_should_update_workflow_flag
        )
        triage_num_main_ideas = st.number_input(
            label="maximum number of main ideas",
            min_value=int(1),
            max_value=int(12),  # arbitrary, is there an empirical upper bound?
            value=INITIAL_NUM_MAIN_IDEAS,
            step=int(1),
            key="num_main_ideas",
            on_change=set_should_update_workflow_flag
        )
    with st.container(border=True):
        st.header("pdf retriever")
        st.file_uploader(
            label="upload pdf files",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_paths",
            help="Upload relevant pdf files.",
            on_change=set_should_update_workflow_flag
        )
        pdf_chunk_size = st.number_input(
            label="chunk size",
            min_value=CHUNK_SIZE_MIN_VALUE,
            max_value=CHUNK_SIZE_MAX_VALUE,
            value=INITIAL_CHUNK_SIZE,
            step=CHUNK_SIZE_STEP,
            key="pdf_chunk_size",
            on_change=set_should_update_workflow_flag
        )
        pdf_chunk_overlap = st.number_input(
            label="chunk overlap",
            min_value=CHUNK_OVERLAP_MIN_VALUE,
            max_value=CHUNK_OVERLAP_MAX_VALUE,
            value=INITIAL_CHUNK_OVERLAP,
            step=CHUNK_OVERLAP_STEP,
            key="pdf_chunk_overlap",
            on_change=set_should_update_workflow_flag
        )
        pdf_k_results = st.number_input(
            label="k results",
            min_value=K_RESULTS_MIN_VALUE,
            max_value=K_RESULTS_MAX_VALUE,
            value=INITIAL_K_RESULTS,
            step=K_RESULTS_STEP,
            key="pdf_k_results",
            on_change=set_should_update_workflow_flag
        )

    with st.container(border=True):
        st.header("url retriever")
        # TODO: dynamic text input widget
        # for now, 10 inputs are provided with empty strings;
        # when any of the inputs are modified, a callback gathers the text from all inputs and assigns to session_state['urls']
        st.text_input(
            label="url",
            value="",
            key="url_input_0",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_1",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_2",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_3",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_4",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_5",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_6",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_7",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_8",
            on_change=set_should_update_workflow_flag
        )
        st.text_input(
            label="url",
            value="",
            key="url_input_9",
            on_change=set_should_update_workflow_flag
        )
        url_chunk_size = st.number_input(
            label="chunk size",
            min_value=CHUNK_SIZE_MIN_VALUE,
            max_value=CHUNK_SIZE_MAX_VALUE,
            value=INITIAL_CHUNK_SIZE,
            step=CHUNK_SIZE_STEP,
            key="url_chunk_size",
            on_change=set_should_update_workflow_flag
        )
        url_chunk_overlap = st.number_input(
            label="chunk overlap",
            min_value=CHUNK_OVERLAP_MIN_VALUE,
            max_value=CHUNK_OVERLAP_MAX_VALUE,
            value=INITIAL_CHUNK_OVERLAP,
            step=CHUNK_OVERLAP_STEP,
            key="url_chunk_overlap",
            on_change=set_should_update_workflow_flag
        )
        url_k_results = st.number_input(
            label="k results",
            min_value=K_RESULTS_MIN_VALUE,
            max_value=K_RESULTS_MAX_VALUE,
            value=INITIAL_K_RESULTS,
            step=K_RESULTS_STEP,
            key="url_k_results",
            on_change=set_should_update_workflow_flag
        )

    with st.container(border=True):
        st.header("arxiv retriever")
        arxiv_use_search = st.checkbox(
            label="arxiv search",
            value=INITIAL_USE_ARXIV_SEARCH,
            key="arxiv_search",
            on_change=set_should_update_workflow_flag
        )
        arxiv_max_docs = st.number_input(
            label="max documents to load",
            min_value=int(1),
            max_value=int(12),
            value=INITIAL_ARXIV_MAX_DOCS_TO_LOAD,
            step=int(1),
            key="arxiv_max_docs_to_load",
            on_change=set_should_update_workflow_flag
        )
        arxiv_chunk_size = st.number_input(
            label="chunk size",
            min_value=CHUNK_SIZE_MIN_VALUE,
            max_value=CHUNK_SIZE_MAX_VALUE,
            value=INITIAL_CHUNK_SIZE,
            step=CHUNK_SIZE_STEP,
            key="arxiv_chunk_size",
            on_change=set_should_update_workflow_flag
        )
        arxiv_chunk_overlap = st.number_input(
            label="chunk overlap",
            min_value=CHUNK_OVERLAP_MIN_VALUE,
            max_value=CHUNK_OVERLAP_MAX_VALUE,
            value=INITIAL_CHUNK_OVERLAP,
            step=CHUNK_OVERLAP_STEP,
            key="arxiv_chunk_overlap",
            on_change=set_should_update_workflow_flag
        )
        arxiv_k_results = st.number_input(
            label="k results",
            min_value=K_RESULTS_MIN_VALUE,
            max_value=K_RESULTS_MAX_VALUE,
            value=INITIAL_K_RESULTS,
            step=K_RESULTS_STEP,
            key="arxiv_k_results",
            on_change=set_should_update_workflow_flag
        )

    with st.container(border=True):
        st.header("vector store db")
        st.multiselect(
            label="collections from database",
            options=COLLECTION_NAMES,
            default=None,  # initialize to be empty,
            key="vdbs_collection_names",
            help="Select a value of multiple values from the list to be retrieved from the database.",
            on_change=set_should_update_workflow_flag
        )
        st.number_input(
            label="k results",
            min_value=K_RESULTS_MIN_VALUE,
            max_value=K_RESULTS_MAX_VALUE,
            value=INITIAL_K_RESULTS,
            step=K_RESULTS_STEP,
            key="vdbs_k_results",
            on_change=set_should_update_workflow_flag
        )


    with st.container(border=True):
        st.header("grader")
        research_context_grader_llm_name = st.selectbox(
            label="context grader model",
            options=[mdl.generic_name for mdl in GRADER_LLMS],
            index=0,
            key="research_context_grader_llm_name",
            on_change=set_should_update_workflow_flag
        )
        research_context_grader_temperature = st.number_input(
            label="temperature",
            min_value=LLM_TEMPERATURE_MIN_VALUE,
            max_value=LLM_TEMPERATURE_MAX_VALUE,
            value=INITIAL_LLM_TEMPERATURE,
            step=LLM_TEMPERATURE_STEP,
            key="research_context_grader_temperature",
            on_change=set_should_update_workflow_flag
        )
        research_context_grader_max_tokens = st.number_input(
            label="maximum number of tokens",
            min_value=LLM_MAX_TOKENS_MIN_VALUE,
            max_value=LLM_MAX_TOKENS_MAX_VALUE,
            value=INITIAL_LLM_MAX_TOKENS,
            step=LLM_MAX_TOKENS_STEP,
            key="research_context_grader_max_tokens",
            on_change=set_should_update_workflow_flag
        )
    with st.container(border=True):
        st.header("Summarizer")
        research_context_summarizer_llm_name = st.selectbox(
            label="context summarizer model",
            options=[mdl.generic_name for mdl in SUMMARIZER_LLMS],
            index=0,
            key="research_context_summarizer_llm_name",
            on_change=set_should_update_workflow_flag
        )
        research_context_summarizer_temperature = st.number_input(
            label="temperature",
            min_value=LLM_TEMPERATURE_MIN_VALUE,
            max_value=LLM_TEMPERATURE_MAX_VALUE,
            value=INITIAL_LLM_TEMPERATURE,
            step=LLM_TEMPERATURE_STEP,
            key="research_context_summarizer_temperature",
            on_change=set_should_update_workflow_flag
        )
        research_context_summarizer_max_tokens = st.number_input(
            label="maximum number of tokens",
            min_value=LLM_MAX_TOKENS_MIN_VALUE,
            max_value=LLM_MAX_TOKENS_MAX_VALUE,
            value=INITIAL_LLM_MAX_TOKENS,
            step=LLM_MAX_TOKENS_STEP,
            key="research_context_summarizer_max_tokens",
            on_change=set_should_update_workflow_flag
        )


# Initialize the chat and workflow if they haven't been created yet
#--memory = MemorySaver()
if not st.session_state["workflow"]:
    update_chat_workflow()
graph = st.session_state["workflow"].compile()  # checkpointer=memory

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Q&A Chat Space
context_documents = []  # filtered docs in the summarizer
if user_prompt := st.chat_input("What would you like to learn about today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display the user message in chat message container
    st.chat_message("user").markdown(user_prompt)
    # if any of the user interface parameters have changes, the state "should_update_workflow"
    # should be True and the workflow will need to be re-built before invoking the graph.
    if st.session_state["should_update_workflow"]:
        update_chat_workflow()
        graph = st.session_state["workflow"].compile()  # checkpointer=memory
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
                if key == ResearchContextGrader.NAME:
                    if value.get("calling_agent") == PdfContextRetriever.NAME:
                        context_documents.extend(value.get("pdf_context", []))
                    if value.get("calling_agent") == WebContextRetriever.NAME:
                        context_documents.extend(value.get("web_context", []))
                    if value.get("calling_agent") == ArxivContextRetriever.NAME:
                        context_documents.extend(value.get("arxiv_context", []))
                    if value.get("calling_agent") == VdbsContextRetriever.NAME:
                        context_documents.extend(value.get("vdbs_context", []))

        # output complete; append the last message to the session_state and show in web interface as markdown
        st.session_state.messages.append({"role": "assistant", "content": message.content})
        st.chat_message("assistant").markdown(message.content)
        # output the sources to chat
        for doc in context_documents:
            message = "\n\n".join(
                [
                    "source:" + str(doc.metadata.get("source", "")),
                    doc.page_content
                ]
            )
            st.chat_message("assistant").markdown(message)
    except Exception as e:
        logging.error(e)
        st.chat_message("assistant").markdown("An error has occured. please try again.")
