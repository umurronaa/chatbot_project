# evaluate_models.py
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics import classification_report
from dotenv import load_dotenv

load_dotenv("credentials.env")

# ---------------------------
# LOAD TEST SET
# ---------------------------
test_df = pd.read_csv("test_set.csv")

# ---------------------------
# LOAD VECTOR DB (TRAIN ONLY)
# ---------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    persist_directory="./gezi_db_train",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

# ---------------------------
# EVALUATION
# ---------------------------
y_true = []
y_pred = []

for _, row in test_df.iterrows():
    question = row["question"]
    true_intent = row["intent"]

    docs = retriever.invoke(question)
    predicted_intent = docs[0].metadata["intent"]

    y_true.append(true_intent)
    y_pred.append(predicted_intent)

# ---------------------------
# METRICS
# ---------------------------
print("\n📊 Classification Report\n")
print(classification_report(y_true, y_pred))
