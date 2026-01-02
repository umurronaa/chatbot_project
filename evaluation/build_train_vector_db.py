# build_vector_db.py
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv("credentials.env")

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_excel("data/avrupa_gezi_rehberi_1000.xlsx")
df.columns = df.columns.str.strip().str.lower()

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["intent"],
    random_state=42
)

# ---------------------------
# BUILD DOCUMENTS (TRAIN ONLY)
# ---------------------------
documents = []
for _, row in train_df.iterrows():
    documents.append(
        Document(
            page_content=row["question"],
            metadata={
                "intent": row["intent"],
                "answer": row["answer"]
            }
        )
    )

# ---------------------------
# VECTOR DB
# ---------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./gezi_db_train"
)

vector_db.persist()

# ---------------------------
# SAVE TEST SET
# ---------------------------
test_df.to_csv("test_set.csv", index=False)

print("Vector DB created with TRAIN set only")
print("Test set saved as test_set.csv")
