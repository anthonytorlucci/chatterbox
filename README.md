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


## research notes to vector databases
Collecting information from the web, arxiv, and pdf documents will ususally provide the context necessary to answer a question.
