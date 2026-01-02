import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

load_dotenv("credentials.env")

df = pd.read_excel("data/avrupa_gezi_rehberi_1000.xlsx")

loader = DataFrameLoader(df, page_content_column="Question")
our_documents = loader.load()

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

google_vector_db = Chroma.from_documents(
    documents=our_documents,
    embedding=gemini_embeddings,
    persist_directory="./gezi_db_google"
)

google_retriever = google_vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6}
)

llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

template = """Sen uzman bir Avrupa gezi danışmanısın.
Bağlamdaki bilgilere sadık kalarak cevap ver.
Yeni bilgi ekleme, yorum yapma.

Bağlam:
{context}

Soru:
{question}

Cevap:
"""

prompt = ChatPromptTemplate.from_template(template)

google_rag_chain = (
    {"context": google_retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm_google
    | StrOutputParser()
)

print("--- Gemini 2.5-Flash RESULT ---")
print(google_rag_chain.invoke("Hoşca kal"))