import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DATA_PATH = "data/sdg.csv"
VECTOR_STORE_PATH = "vectorstore/faiss_index"

# --------------- STEP 1: Load & Analyze CSV ---------------- #
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print("---- Dataset Preview ----")
    print(df.head(), "\n")
    print("---- Dataset Info ----")
    print(df.info(), "\n")
    print("---- Null Values ----")
    print(df.isnull().sum(), "\n")
    return df

# --------------- STEP 2: Convert to Documents -------------- #
def convert_to_documents(df):
    documents = []
    for i, row in df.iterrows():
        metadata = {
            "country": row["Country Name"],
            "code": row["Country Code"],
            "indicator": row["Indicator Name"],
            "indicator_code": row["Indicator Code"]
        }

        for year in range(1990, 2019):
            year_str = str(year)
            if pd.notnull(row.get(year_str)):
                content = f"{row['Country Name']} - {row['Indicator Name']} in {year_str}: {row[year_str]}"
                documents.append({"content": content, "metadata": metadata})
    
    print(f"Converted {len(documents)} rows to Document format.")
    return documents

# --------------- STEP 3: Build FAISS Vector Store ---------- #
def build_vector_store(documents):
    print("Creating new FAISS index...")

    # Load the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create embeddings
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save index and documents
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    faiss.write_index(index, f"{VECTOR_STORE_PATH}.index")
    
    with open(f"{VECTOR_STORE_PATH}_docs.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    print(f"Saved FAISS index to: {VECTOR_STORE_PATH}")
    return index, documents

# --------------- STEP 4: Load Vector Store ----------------- #
def load_vector_store():
    index = faiss.read_index(f"{VECTOR_STORE_PATH}.index")
    with open(f"{VECTOR_STORE_PATH}_docs.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

# --------------- MAIN EXECUTION ---------------------------- #
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    documents = convert_to_documents(df)

    if not os.path.exists(f"{VECTOR_STORE_PATH}.index"):
        index, documents = build_vector_store(documents)
    else:
        print("Loading existing FAISS index...")
        index, documents = load_vector_store()

    # Optional: Run a sample query
    query = "What is the electricity access in Arab World in 2015?"
    
    # Load model for query
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    
    # Search
    D, I = index.search(query_embedding.astype('float32'), 3)
    
    print("\nTop 3 relevant results:")
    for i, idx in enumerate(I[0]):
        print(f"{i+1}. {documents[idx]['content']}")
        print(f"   Score: {D[0][i]:.4f}")
