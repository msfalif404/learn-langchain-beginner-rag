import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load Environment Variable
load_dotenv()

# Setup Embedding & Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
)

# Setup LLM
llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Gunakan dokumen berikut untuk menjawab pertanyaan: {context}\n\nJangan berikan informasi lain jika tidak tahu."),
    ("human", "{input}")
])

# Combine documents using the prompt
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Build Retrieval Chain
qa_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=combine_docs_chain
)

# Stateless QA
query_1 = "Apa penyebab kecemasan menurut paper tersebut? Berikan nomor untuk setiap penyebab."
res1 = qa_chain.invoke({"input": query_1})
print("\n--- Jawaban 1 ---")
print(res1['answer'])

query_2 = "Jelaskan lebih detail untuk penyebab nomor 2!"
res2 = qa_chain.invoke({'input': query_2})
print("\n--- Jawaban 2 ---")
print(res2['answer'])