"""
Frontend streamlit app
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

# third party
import streamlit as st
from pydantic import (
    AnyUrl
)

# langchain
from langchain_core.messages import HumanMessage, AIMessage

# langgraph

# local
from chatterbox.language_models import LargeLanguageModelConfig, LargeLanguageModelsEnum
from chatterbox.nodes.simple_chat import SimpleChat
from chatterbox.workflows import (
    build_simple_chat_graph
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

# ---- global variables ----
# Configure logging
logging.basicConfig(
    #filename="Path(__file__).parent.joinpath("logs", "chat.log",  # Log to a file
    level=logging.INFO,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s", # Customize the log message format
    #filemode="w" # Overwrite log file on each run. Use 'a' to append.
)

LLM_MODELS = [llm for llm in LargeLanguageModelsEnum]

# ---- streamlit app configuration and state
# streamlit page configuration
st.title("Chatterbox")

if "simple_chat_node" not in st.session_state:
    st.session_state["simple_chat_node"] = None
if "workflow" not in st.session_state:
    st.session_state["workflow"] = None

# ---- callbacks ----
def update_chat_workflow():
    """Callback to update the chat node and workflow when settings change"""
    llm_enum = [mdl for mdl in LargeLanguageModelsEnum if mdl.generic_name == st.session_state["chat_agent_llm_name"]][0]
    logging.info(f""".... update_chat_workflow called.\n""")
    logging.info(f""".... language model: {llm_enum}""")
    logging.info(f""".... use search tool: {st.session_state["chat_agent_use_search"]}""")
    match llm_enum.company:
        case "OpenAI":
            llm_api_key = OPENAI_API_KEY
        case "Anthropic":
            llm_api_key = ANTHROPIC_API_KEY
        case "Fireworks":
            llm_api_key = FIREWORKS_API_KEY
        case _:
            llm_api_key = ""

    st.session_state["simple_chat_node"] = SimpleChat(
        model_config=LargeLanguageModelConfig(
            id=llm_enum,
            api_key=llm_api_key,
            temperature=st.session_state["chat_agent_temperature"],
            max_tokens=st.session_state["chat_agent_max_tokens"]
        ),
        use_search=st.session_state["chat_agent_use_search"]
    )
    st.session_state["workflow"] = build_simple_chat_graph(
        simple_chat_node=st.session_state["simple_chat_node"]
    )


# ---- button box ----
with st.sidebar:
    # chat agent config
    with st.container(border=True):
        llm_name = st.selectbox(
            label="select model",
            options=[mdl.generic_name for mdl in LLM_MODELS],
            index=[i for i,llm in enumerate(LLM_MODELS) if llm == LargeLanguageModelsEnum.OLLAMA_LLAMA_32_3B][0],
            #index=0,
            key="chat_agent_llm_name",
            on_change=update_chat_workflow
        )
        temperature = st.number_input(
            label="temperature",
            min_value=float(0.0),
            max_value=float(1.0),
            value=float(0.5),
            step=float(0.01),
            key="chat_agent_temperature",
            on_change=update_chat_workflow
        )
        max_tokens = st.number_input(
            label="maximum number of tokens",
            min_value=int(1000),
            max_value=int(128_000),
            value=int(1000),
            step=int(1000),
            key="chat_agent_max_tokens",
            on_change=update_chat_workflow
        )
        use_search = st.checkbox(
            label="use search tool",
            value=False,
            key="chat_agent_use_search",
            on_change=update_chat_workflow
        )

    st.divider()
    # TODO: convert session_state.messages to markdown; open a file dialog to save the markdown file
    st.button(
        label="Export Markdown",
        key="export_markdown_button",
        # TODO: on_click=None,
        disabled=False,
        use_container_width=False,
    )

# Initialize the chat node and workflow if they haven't been created yet
if st.session_state["simple_chat_node"] is None:
    update_chat_workflow()

graph = st.session_state["workflow"].compile()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Q&A Chat Space
if user_prompt := st.chat_input("What would you like to learn about today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display the user message in chat message container
    st.chat_message("user").markdown(user_prompt)
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

        # output complete; append the last message to the session_state and show in web interface as markdown
        st.session_state.messages.append({"role": "assistant", "content": message.content})
        st.chat_message("assistant").markdown(message.content)
    except Exception as e:
        logging.error(e)
        st.chat_message("assistant").markdown("An error has occured. please try again.")
