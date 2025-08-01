from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from llm import *

embeddings = OllamaEmbeddings(model="llama3.2:1b")
url=""
api_key=""

question = input("Enter your question: ")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=url,
    api_key=api_key,
    collection_name="cricket",
)

response = qdrant.similarity_search(
    question,
    k=5
)

# Format the context from the retrieved documents
context = "\n\n".join([doc.page_content for doc in response])

prompt = f"""
Question: {question}

Context: {context}

Please provide a summary based on the provided content.
"""

completion_prompt(prompt)
