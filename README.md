# chatterbox

> [!WARNING]
> At this time, this is a personal project and not intended for distribution.

My primary use for generative ai leveraging large language models is for scientific research and code development.
While today's llm models are quite adept at solving most problems, I often would like to feed research articles and/or open source projects to the llm for additional context. Many of those research articles are likely outside the scope of the llm's training data.

Chatterbox is a collection langgraph workflows (graphs) made up of components (nodes, conditional edges, and utilities). Each workflow has an associated frontend web application built with Streamlit.

Run the app to launch stremlit chat (requires an .env file with API KEYS)
```zsh
uv run python app.py --chat
```

Or
```zsh
uv run --env-file .env -- python st_chat.py
```

## large language models
One of the objectives of the project was to explore different large language models for different agents. Chatterbox does this by providing a function `get_llm_model` that returns a [BaseChatModel](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel) for each llm defined in the `LargeLanguageModelsEnum`.

See [language_models.py](./chatterbox/language_models.py)

### requirements for using large language models
To use the language models provided by Anthropic, OpenAI, or Fireworks, requires an api key which must be stored in the `.env` file.

## research notes to vector databases
Collecting information from the web, arxiv, and pdf documents will ususally provide the context necessary to answer a question, if you get the chunking right and have enough context in each document.

Another option is to build a collection of [document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)s from research notes. To do this, I write my research notes in latex files (.tex) and use the ... to load and save to a [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) database.


## Chat
The chat app is the simplest application ...

To run,

```zsh
uv run python app.py --chat
```

## Research Context Summary
The objective of this graph is to summarize all relevant documents to the input research prompt.

### agents (nodes)

#### triage

#### research context supervisor

#### pdf context retriever

#### web context retriever

#### arxiv context retriever

#### vdbs context retriever

#### research context grader

#### research context summarizer

## discussion and future work
<!--
1. moving away from vector similarity search and toward graph rag or light rag
2. plan and execute subgraph to generate report rather than just summarize; check out [gpt-researcher](https://gptr.dev)
-->
