import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import joblib

load_dotenv("credentials.env")
api_key_openai = os.getenv("OPENAI_API_KEY")

df = pd.read_excel("avrupa_gezi_rehberi_1000.xlsx")

loader = DataFrameLoader(df, page_content_column="Question")
our_documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

openai_vector_db = Chroma.from_documents(documents=our_documents,
                                  embedding=embeddings,
                                  persist_directory="./gezi_db_openai")

retriever = openai_vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":6})

llm_openai = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
output_parser = StrOutputParser()

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

openai_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_openai
    | output_parser
)

question = "Roma'da ne yenir?"
openai_answer = openai_rag_chain.invoke(question)

print("--- OpenAI (GPT-4.1-mini) Result ---")
print(openai_answer)