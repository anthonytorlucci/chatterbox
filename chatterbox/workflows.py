"""Functions and classes to generate various supported workflows."""

# standard lib
from typing import (
    Sequence,
    Annotated,
    TypedDict
)

# third party

# langchain
from langchain_core.messages import (
    BaseMessage,
)
# langgraph
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START

# local
from chatterbox.nodes.simple_chat import SimpleChat


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
