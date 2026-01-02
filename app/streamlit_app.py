import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv("credentials.env")

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="European Travel Guide - Model Comparison",
    layout="wide"
)

st.title("European Travel Guide RAG Comparison")
st.markdown("GPT-4.1-mini vs Gemini-2.5-Flash")

# --------------------------------------------------
# INIT MODELS & DB (CACHED)
# --------------------------------------------------
@st.cache_resource
def init_models():

    # --- OpenAI ---
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    openai_db = Chroma(
        persist_directory="./gezi_db_openai",
        embedding_function=openai_embeddings
    )

    openai_retriever = openai_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    openai_llm = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0
    )

    # --- Google / Gemini ---
    google_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    google_db = Chroma(
        persist_directory="./gezi_db_google",
        embedding_function=google_embeddings
    )

    google_retriever = google_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    google_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    return openai_retriever, openai_llm, google_retriever, google_llm


openai_retriever, openai_llm, google_retriever, google_llm = init_models()

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
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

# --------------------------------------------------
# TRUE RAG CHAINS (AYNI _model.py GİBİ)
# --------------------------------------------------
openai_chain = (
    {"context": openai_retriever, "question": RunnablePassthrough()}
    | prompt
    | openai_llm
    | StrOutputParser()
)

google_chain = (
    {"context": google_retriever, "question": RunnablePassthrough()}
    | prompt
    | google_llm
    | StrOutputParser()
)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
user_query = st.text_input(
    "Gezi hakkında bir soru sorun (Örn: Roma'da ne yenir?)"
)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if user_query:

    col1, col2 = st.columns(2)

    # --- OpenAI ---
    with col1:
        st.subheader("OpenAI (GPT-4.1-mini)")
        t0 = time.time()
        result_oa = openai_chain.invoke(user_query)
        dur_oa = time.time() - t0

        st.write(result_oa)
        st.caption(f"Response Time: {dur_oa:.2f}s")

    # --- Gemini ---
    with col2:
        st.subheader("Google (Gemini-2.5-Flash)")
        t1 = time.time()
        result_go = google_chain.invoke(user_query)
        dur_go = time.time() - t1

        st.write(result_go)
        st.caption(f"Response Time: {dur_go:.2f}s")

    # --------------------------------------------------
    # SIDEBAR DEBUG (SADECE GÖRSEL KONTROL)
    # --------------------------------------------------
    with st.sidebar:
        st.header("Retrieved Documents (Debug)")

        docs_oa = openai_retriever.invoke(user_query)
        docs_go = google_retriever.invoke(user_query)

        st.write(f"OpenAI Retrieved: {len(docs_oa)} docs")
        st.write(f"Gemini Retrieved: {len(docs_go)} docs")

        with st.expander("OpenAI Raw Docs"):
            for i, d in enumerate(docs_oa, 1):
                st.markdown(f"**Doc {i}:**")
                st.write(d.page_content)

        with st.expander("Gemini Raw Docs"):
            for i, d in enumerate(docs_go, 1):
                st.markdown(f"**Doc {i}:**")
                st.write(d.page_content)
