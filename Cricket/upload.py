from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

embeddings = OllamaEmbeddings(model="llama3.2:1b")

file_path = r"C:\Users\CSESTUDENT\Desktop\cricket\cricket.pdf"
loader = PyPDFLoader(file_path)
data = loader.load_and_split()

url=""
api_key=""


qdrant = QdrantClient(
    url="",
    api_key="",
)

print(qdrant.get_collections())

qdrant = QdrantVectorStore.from_documents(
    data,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="cricket",
)
