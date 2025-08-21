import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load Environment Variable
load_dotenv()

# Setup Embedding & Vectore Store
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

# Setup LLM
llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini")

# History-Aware Retriever Prompt
history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
    ("user", "Berdasarkan percakapan di atas, buatlah pertanyaan yang mandiri untuk mencari informasi yang relevan dengan percakapan ini.")
])

# History-Aware Retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    vector_store.as_retriever(),
    history_aware_retriever_prompt
)

# Prompt Untuk Menggabungkan Dokumen dan Jawaban
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan. Gunakan dokumen berikut untuk menjawab pertanyaan: {context}\n\nRiwayat Percakapan:\n{chat_history}\n\nJangan berikan informasi lain jika tidak tahu."),
    ("human", "{input}")
])

# Chain Untuk Menggabungkan Dokumen
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# Menggabungkan History-Aware Retriever dan combine_docs_chain
# Alur: Rephrasing Query > Retrieval > Generation
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever,
    combine_docs_chain
)

# Inisialisasi Riwayat Chat Kosong
chat_history = []

print("--- Memulai Percakapan ---")

# Pertanyaan 1
query_1 = "Apa penyebab kecemasan menurut dokumen tersebut ? Berikan 3 poin penyebabnya tanpa perlu dijelaskan."
print(f"\nHuman: {query_1}")
res1 = conversational_rag_chain.invoke({
    "input": query_1,
    "chat_history": chat_history
})
print("\n--- Jawaban 1 ---")
print(f"AI:\n {res1['answer']}")
print()

# Tambahkan Pertanyaan dan Jawaban Ke Riwayat Chat
chat_history.append(HumanMessage(content=query_1))
chat_history.append(AIMessage(content=res1['answer']))

# Pertanyaan 2
query_2 = "Jelaskan secara singkat penyebab nomor 1!"
print(f"\nHuman: {query_2}")
res2 = conversational_rag_chain.invoke({
    "input": query_2,
    "chat_history": chat_history
})
print("\n--- Jawaban 2 ---")
print(f"AI:\n {res2['answer']}")
print()

# Tambahkan Pertanyaan dan Jawaban Ke Riwayat Chat
chat_history.append(HumanMessage(content=query_2))
chat_history.append(AIMessage(content=res2['answer']))

# Pertanyaan 3
query_3 = "Dari ketiga model, mana yang memiliki performa paling bagus ?"
print(f"\nHuman: {query_3}")
res3 = conversational_rag_chain.invoke({
    "input": query_3,
    "chat_history": chat_history
})
print("\n--- Jawaban 3 ---")
print(f"AI:\n {res3['answer']}")
print()

# Tambahkan Pertanyaan dan Jawaban Ke Riwayat Chat
chat_history.append(HumanMessage(content=query_3))
chat_history.append(AIMessage(content=res3['answer']))

print()
print("--- Riwayat Percakapan Akhir ---")
for message in chat_history:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")