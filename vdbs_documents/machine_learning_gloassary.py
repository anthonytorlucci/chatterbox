"""
Collection of LangChain documents from https://developers.google.com/machine-learning/glossary.
"""
# standard library
from pathlib import Path
from uuid import uuid4
# third party
# langchain
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# langgraph
# local

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = Chroma(
    collection_name="google_machine_learning_gloassary",
    embedding_function=embeddings,
    persist_directory=str(Path(__file__).parent.joinpath("chroma_research_notes_ollama_emb_db")),  # to save data locally
)
docs = [
    Document(
        page_content="""ablation
    A technique for evaluating the importance of a feature or component by temporarily removing it from a model. You then retrain the model without that feature or component, and if the retrained model performs significantly worse, then the removed feature or component was likely important.

    For example, suppose you train a classification model on 10 features and achieve 88% precision on the test set. To check the importance of the first feature, you can retrain the model using only the nine other features. If the retrained model performs significantly worse (for instance, 55% precision), then the removed feature was probably important. Conversely, if the retrained model performs equally well, then that feature was probably not that important.

    Ablation can also help determine the importance of:
        - Larger components, such as an entire subsystem of a larger ML system
        - Processes or techniques, such as a data preprocessing step

    In both cases, you would observe how the system's performance changes (or doesn't change) after you've removed the component.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""A/B testing
    A statistical way of comparing two (or more) techniques—the A and the B. Typically, the A is an existing technique, and the B is a new technique. A/B testing not only determines which technique performs better but also whether the difference is statistically significant.\n\nA/B testing usually compares a single metric on two techniques; for example, how does model accuracy compare for two techniques? However, A/B testing can also compare any finite number of metrics.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""accelerator chip
A category of specialized hardware components designed to perform key computations needed for deep learning algorithms.

Accelerator chips (or just accelerators, for short) can significantly increase the speed and efficiency of training and inference tasks compared to a general-purpose CPU. They are ideal for training neural networks and similar computationally intensive tasks.

Examples of accelerator chips include:
    - Google's Tensor Processing Units (TPUs) with dedicated hardware for deep learning.
    - NVIDIA's GPUs which, though initially designed for graphics processing, are designed to enable parallel processing, which can significantly increase processing speed.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""accuracy
The number of correct classification predictions divided by the total number of predictions. That is:
$$\\text { Accuracy }=\\frac{\\text { correct predictions }}{\\text { correct predictions }+ \\text { incorfect predictions }}$$

For example, a model that made 40 correct predictions and 10 incorrect predictions would have an accuracy of:
$$\\text { Aсcuracy }=\\frac{40}{40+10}=80 \\%$$

Binary classification provides specific names for the different categories of _correct predictions_ and _incorrect predictions_. So, the accuracy formula for binary classification is as follows:
$$\\text{Accuracy}=\\frac{TP + TN}{TP + TN + FP + FN}$$

where:
    - TP is the number of true positives (correct predictions).
    - TN is the number of true negatives (correct predictions).
    - FP is the number of false positives (incorrect predictions).
    - FN is the number of false negatives (incorrect predictions).

Compare and contrast accuracy with precision and recall.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""action
In reinforcement learning, the mechanism by which the agent transitions between states of the environment. The agent chooses the action by using a policy.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""activation function
A function that enables neural networks to learn nonlinear (complex) relationships between features and the label.

Popular activation functions include:
    - ReLU
    - Sigmoid

The plots of activation functions are never single straight lines.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""active learning
A trainin approach in which the algorithm chooses some of the data it learns from. Active learning is particularly valuable when labeled examples are scarce or expensive to obtain. Instead of blindly seeking a diverse range of labeled examples, an active learning algorithm selectively seeks the particular range of examples it needs for learning.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""AdaGrad
A sophisticated gradient descent algorithm that rescales the gradients of each parameter, effectively giving each parameter an independent learning rate.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""agent
In reinforcement learning, the entity that uses a policy to maximize the expected return gained from transitioning between states of the environment.

More generally, an agent is software that autonomously plans and executes a series of actions in pursuit of a goal, with the ability to adapt to changes in its environment. For example, an LLM-based agent might use an LLM to generate a plan, rather than applying a reinforcement learning policy.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""anomaly detection
The process of identifying outliers. For example, if the mean for a certain feature is 100 with a standard deviation of 10, then anomaly detection should flag a value of 200 as suspicious.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""artificial general intelligence
A non-human mechanism that demonstrates a broad range of problem solving, creativity, and adaptability. For example, a program demonstrating artificial general intelligence could translate text, compose symphonies, and excel at games that have not yet been invented.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""artificial intelligence
A non-human program or model that can solve sophisticated tasks. For example, a program or model that translates text or a program or model that identifies diseases from radiologic images both exhibit artificial intelligence.

Formally, machine learning is a sub-field of artificial intelligence. However, in recent years, some organizations have begun using the terms artificial intelligence and machine learning interchangably.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""attention
A mechanism used in a neural network that indicates the importance of a particular word or part of a word. Attention compresses the amount of information a model needs to predict the next token/word. A typical attention mechanism might consist of a weighted sum over a set of inputs, where the weight for each input is computed by another part of the neural network.

Refer also to self-attention and multi-head self-attention, which are the building blocks of Transformers.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""attribute
Synonym for feature.

In machine learning fairness, attributes often refer to characteristics pertaining to individuals.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""attribute sampling
A tactic for training a decision forest in which each decision tree considers only a random subset of possible features when learning the condition. Generally, a different subset of features is sampled for each node. In contrast, when training a decision tree without attribute sampling, all possible features are considered for each node.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""augmented reality
A technology that superimposes a computer-generated image on a user's view of the real world, thus providing a composite view.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""autoencoder
A system that learns to extract the most important information from the input. Autoencoders are a combination of an encoder and a decoder. Autoencoders rely on the following two-step process:
    1.The encoder maps the input to a (typically) lossy lower-dimensional (intermediate) format.
    2. The decoder builds a lossy version of the original input by mapping the lower-dimensional format to the original higher-dimensional input format.

Autoencoders are trained end-to-end by having the decoder attempt to reconstruct the original input from the encoder's intermediate format as closely as possible. Because the intermediate format is smaller (lower-dimensional) that the original format, the autoencodder is forced to learn what information in the input is essential, and the output won't be perfectly identical to the input.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""automaitic evaluation
Using software to judge the quality of a model's output.

When the model output is relatively straightforward, a script or program can compare the model's output to a goldn response. This type of automatic evaluation is sometimes called programmatic evaluation. Metrics such as ROUGE or BLEU are often useful for programmatic evaluation.

When model output is complex or has no one right answer, a separate ML program called an autorater sometimes performs the automatic evaluation.

Contrast with human evaluation.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""automation bias
When a human decision maker favors recomendations made by an automated decision-making system over information made without automation, even when the automated decision-making system makes errors.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""AutoML
Any automated process for building machine learning models. AutoML can automatically do tasks such as the following:
    - Search for the most appropriate model.
    - Tune hyperparameters.
    - Prepare data (including performing feature engineering).
    - Deploy the resulting model.

AutoML is useful for data scientists because it can save them time and effort in developing machine learning pipelines and improve prediction accuracy. It is also useful to non-experts, by making complicated machine learning tasks more accessible to them.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""autorater evaluation
A hybrid mechanism for judging the quality of a generative AI model's output that combines human evaluation with automatic evaluation. An autorater is an ML model trained on data created by human evaluation. Ideally, an autorater learns to mimic a human evaluator.

Prebuilt autoraters are available, but the best autoraters are fine-tuned spefically to the task you are evaluating.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
    Document(
        page_content="""auto-regressive model
A model that infers a prediction based on its own previous predictions. For example, auto-regressive language models predict the next token based on the previously predicted tokens. All Transformer-based large language models are auto-regressive.

In contrast, GAN-based image models are usually not auto-regressive since they generate an image in a single forward-pass and not iteratively in steps. However, certain image generation models are auto-regressive because they generate an image in steps.
""",
        metadata={"source": "https://developers.google.com/machine-learning/glossary"}
    ),
#     Document(
#         page_content="""
# """,
#         metadata={"source": "https://developers.google.com/machine-learning/glossary"}
#     ),
#     Document(
#         page_content="""
# """,
#         metadata={"source": "https://developers.google.com/machine-learning/glossary"}
#     ),
#     Document(
#         page_content="""
# """,
#         metadata={"source": "https://developers.google.com/machine-learning/glossary"}
#     ),
#     Document(
#         page_content="""
# """,
#         metadata={"source": "https://developers.google.com/machine-learning/glossary"}
#     ),
]

uuids = [str(uuid4()) for _ in range(len(docs))]

vector_store.add_documents(documents=docs, ids=uuids)
