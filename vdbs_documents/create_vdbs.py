# standard library
from pathlib import Path
from uuid import uuid4
# third party
# langchain
from langchain_core.documents import Document
from langchain_text_splitters.latex import LatexTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# langgraph
# local

embeddings = OllamaEmbeddings(model="llama3.2")

# tex = Path(__file__).parent.joinpath("medium_generative_driven_design.tex")
# with open(tex, 'r') as tex_file:
#     tex_content = tex_file.read()

# text_splitter = LatexTextSplitter()
# split_tex = text_splitter.split_text(tex_content)
# documents = [
#     Document(
#         page_content=t,
#         metadata={"source": "https://towardsdatascience.com/gdd-generative-driven-design-0c948fb9a735"}
#     ) for t in split_tex
# ]

# vector_store = Chroma(
#     collection_name="medium_generative_driven_design",
#     embedding_function=embeddings,
#     persist_directory=str(Path(__file__).parent.joinpath("chroma_research_notes_ollama_emb_db")),  # to save data locally
# )
# vector_store.add_documents(documents)

tex = Path(__file__).parent.joinpath("medium_neural_networks_are_fundamentally_Bayesian.tex")
with open(tex, 'r') as tex_file:
    tex_content = tex_file.read()

text_splitter = LatexTextSplitter()
split_tex = text_splitter.split_text(tex_content)
documents = [
    Document(
        page_content=t,
        metadata={"source": "https://towardsdatascience.com/neural-networks-are-fundamentally-bayesian-bee9a172fad8"}
    ) for t in split_tex
]

vector_store = Chroma(
    collection_name="medium_neural_networks_are_fundamentally_bayesian",
    embedding_function=embeddings,
    persist_directory=str(Path(__file__).parent.joinpath("chroma_research_notes_ollama_emb_db")),  # to save data locally
)
vector_store.add_documents(documents)
