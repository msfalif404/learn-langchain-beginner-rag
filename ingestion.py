import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Memuat Dokumen")
loader = PyPDFLoader("./paper.pdf")
document = loader.load()

print("Memproses Dokumen")
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"Created {len(texts)} chunks")

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

try:
    print("Menyimpan Informasi PDF ke Pinecone")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME"))
    print("Selesai Menyimpan Informasi")
except Exception as e:
    print("ERROR")
    print(e)