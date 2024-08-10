# End-to-End RAG App with OpenAI and Qdrant

This repository contains an end-to-end application using Retrieval-Augmented Generation (RAG) with LangChain, OpenAI, and Qdrant. The app allows users to query a PDF document, retrieving relevant text snippets using Qdrant's vector search, and then generating concise answers with the help of OpenAI's language models.

## Table of Contents

- [What is Qdrant?](#what-is-qdrant)
- [Concept of Retrieval-Augmented Generation (RAG)](#concept-of-retrieval-augmented-generation-rag)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## What is Qdrant?

Qdrant is an open-source vector search engine designed to handle and query high-dimensional vector data efficiently. It is used to perform similarity searches on large sets of data, enabling applications like semantic search, recommendation systems, and more. Qdrant works by indexing vector embeddings, which are numerical representations of data (such as text chunks), allowing fast retrieval of the most relevant results based on vector similarity.

## Concept of Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a method that combines retrieval-based techniques with generative models to improve the quality of answers generated from textual data. The process involves:

1. **Retrieval**: Using a vector search engine (like Qdrant) to retrieve relevant chunks of text or documents from a large corpus based on a given query. This step helps in identifying the most pertinent information related to the query.

2. **Generation**: Feeding the retrieved text chunks into a generative model (like OpenAI's GPT) to generate a coherent and contextually relevant response. The generative model uses the retrieved context to produce answers that are more accurate and specific to the query.

RAG effectively bridges the gap between information retrieval and text generation, providing a more effective way to answer questions based on large-scale textual data.

## Installation

To run this project, you'll need to install the required Python packages. Use the following commands to set up your environment:

```bash
pip install langchain openai tiktoken qdrant-client langchain_openai pypdf langchainhub langchain_community langchain-groq
```

## Project Structure

- **`OpenAI-End-to-End-App-with-RAG.ipynb`**: The main Jupyter Notebook containing the code for the RAG app.
- **`thousandsplendidsuns.pdf`**: A sample PDF document used for testing the app (replace with your own document).
- **`README.md`**: This file.

## Usage

### 1. Load the Document

The app starts by loading a PDF document using the `PyPDFLoader`. The content is split into manageable chunks using `RecursiveCharacterTextSplitter`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/thousandsplendidsuns.pdf")
documents = loader.load()
```

### 2. Split the Document into Chunks

To efficiently process the document, it's split into chunks of 1000 characters with a 200-character overlap.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
```

### 3. Generate Embeddings and Store them in Qdrant

Using OpenAI's embedding model, embeddings are generated for each chunk of text and stored in a Qdrant vector database.

```python
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)

qdrant = Qdrant.from_documents(
    texts,
    embedding=embeddings_model,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_api_key,
    collection_name="llm-app"
)
```

### 4. Retrieve Relevant Documents

The app uses Qdrant's vector search to retrieve the most relevant chunks of text for a given query.

```python
retriever = qdrant.as_retriever(search_kwargs={"k": 5})
query = "Who was Marium?"
docs = retriever.get_relevant_documents(query)
```

### 5. Generate Answers with OpenAI's GPT Model

A prompt is created using the retrieved documents, and OpenAI's GPT model generates a concise answer.

```python
prompt = PromptTemplate(
    template="""...""",  # Template provided in the notebook
    input_variables=["context", "question"]
)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
```

### 6. Run the RAG Chain

The RAG chain is invoked with a user query to generate the final answer.

```python
rag_chain.invoke("Who was Marium?")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
