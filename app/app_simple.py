import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="🌍 SDG Chatbot", layout="wide")
st.title("🌍 Sustainable Development Goals (SDG) Chatbot")
st.markdown("Ask anything about SDGs, indicators, or global development 📊")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    vectorstore_path = "vectorstore/faiss_index"
    if os.path.exists(f"{vectorstore_path}.index") and os.path.exists(f"{vectorstore_path}_docs.pkl"):
        index = faiss.read_index(f"{vectorstore_path}.index")
        with open(f"{vectorstore_path}_docs.pkl", "rb") as f:
            documents = pickle.load(f)
        return index, documents
    else:
        st.error("Vector store not found. Please run the RAG pipeline first.")
        return None, None

# Load local LLM with HuggingFacePipeline
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return pipe

# Load sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize components
index, documents = load_vectorstore()
llm = load_llm()
embedding_model = load_embedding_model()

if index is not None and documents is not None:
    # Chat input
    query = st.text_input("Enter your question here 👇")

    if query:
        with st.spinner("Thinking..."):
            # Create query embedding
            query_embedding = embedding_model.encode([query])
            
            # Search for similar documents
            D, I = index.search(query_embedding.astype('float32'), 5)
            
            # Get relevant context
            relevant_docs = [documents[idx] for idx in I[0]]
            context = "\n".join([doc["content"] for doc in relevant_docs])
            
            # Create prompt for LLM
            prompt = f"Based on the following information about Sustainable Development Goals, answer this question: {query}\n\nInformation:\n{context}\n\nAnswer:"
            
            # Generate answer
            result = llm(prompt, max_length=256, do_sample=True)[0]['generated_text']
            
            st.success("Here's the answer:")
            st.write(result)
            
            # Show relevant context
            with st.expander("📚 Relevant Data Used"):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**{i+1}.** {doc['content']}")
                    st.write(f"   *Country: {doc['metadata']['country']}, Indicator: {doc['metadata']['indicator']}*")
                    st.write("---")

else:
    st.error("""
    ## Setup Required
    
    The vector store hasn't been created yet. Please run the following command first:
    
    ```bash
    python chatbot/rag_pipeline_simple.py
    ```
    
    This will create the necessary FAISS index and document store.
    """)

# Footer
st.markdown("---")
st.markdown("🛠️ Built using Sentence Transformers, FAISS, HuggingFace Transformers, and Streamlit")
