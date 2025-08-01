from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from llm import completion_prompt         

embeddings = OllamaEmbeddings(model="llama3.2:1b")


url=""
api_key=""

question = input("Enter your question: ")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="bitcoin",
    url=url,
    api_key=api_key,
)

response = qdrant.similarity_search(
    question, k=2
)
prompt = f"""
Question: {question}

context : {response}

you are a helpful assistant that can answer questions about the context provided.
"""

print("prompt: ",prompt)
print(completion_prompt(prompt))

