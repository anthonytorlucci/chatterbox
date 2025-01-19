"""
simple_chat

Simple chatbot agent with optional search tool.
"""

# standard lib

# third party

# langchain
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    #MessagesPlaceholder
)
from langchain_community.tools import TavilySearchResults

# langgraph

# local
from chatterbox.language_models import (
    LargeLanguageModelConfig,
    get_llm_model
)
from chatterbox.researcher_interface import (
    ResearcherInterface
)



### subclass ResearcherInterface

class SimpleChat(ResearcherInterface):
    """A simple chat interface that provides basic conversational responses.

    This class implements the ResearcherInterface to create a straightforward chatbot
    that can optionally use web search capabilities. It uses a language model to
    generate responses to user prompts.

    Attributes:
        NAME (str): The identifier name for this chat implementation ("simple_chat")
        _use_search (bool): Flag indicating whether web search capability is enabled
        simple_chatter: A chain combining the chat prompt template with the language model

    Args:
        model_config (LargeLanguageModelConfig): Configuration for the language model
        use_search (bool, optional): Whether to enable web search capabilities. Defaults to False.

    Example:
        ```python
        model_config = LargeLanguageModelConfig(...)
        chat = SimpleChat(model_config, use_search=True)
        response = chat({"messages": [{"content": "Hello, how are you?"}]})
        ```

    Notes:
        - When search is enabled, it uses the Tavily search API for web searches
        - The chat uses a basic system prompt that identifies itself as a helpful assistant
    """
    NAME = "simple_chat"
    def __init__(
        self,
        model_config: LargeLanguageModelConfig,
    ):
        llm_chat = get_llm_model(model_config=model_config)

        # Prompt
        sys_prompt=PromptTemplate(
            input_variables=[],
            template="""You are a helpful assistant.
            """
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

        human_prompt: PromptTemplate = PromptTemplate(
            input_variables=["input_prompt"],
            template="""Write a well-posed response to the user prompt:
            {input_prompt}
            """
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

        agent_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])


        self.simple_chatter = agent_prompt | llm_chat

    def __call__(self, state: dict):
        """
        Respond to user prompt.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): add to messages.
        """
        messages = state.get("messages", "")
        message = messages[-1].content
        if message:
            #print(message)
            response = self.simple_chatter.invoke(
                input={
                    "input_prompt": message,
                },
                #config=self._runnable_config
                # config=RunnableConfig(
                #     configurable={"thread_id": "0"}
                # )
            )
            return {"messages": [response]}
        else:
            raise ValueError("No messages found.")
