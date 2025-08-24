import streamlit as st
import faiss
import pickle
import os
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="🌍 SDG Chatbot", layout="wide")
st.title("🌍 Sustainable Development Goals (SDG) Chatbot")
st.markdown("Ask anything about SDGs, indicators, or global development 📊")

# Function to ensure vector store exists
def ensure_vectorstore():
    """Build vector store if it doesn't exist"""
    if not os.path.exists("vectorstore/faiss_index.index"):
        with st.spinner("🔄 Building vector store... This may take a few minutes on first run."):
            try:
                # Run the RAG pipeline
                result = subprocess.run([
                    sys.executable, "chatbot/rag_pipeline_simple.py"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    st.success("✅ Vector store built successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed to build vector store: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                st.error("❌ Vector store build timed out. Please try again.")
                return False
            except Exception as e:
                st.error(f"❌ Error building vector store: {str(e)}")
                return False
    return True

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    vectorstore_path = "vectorstore/faiss_index"
    if os.path.exists(f"{vectorstore_path}.index") and os.path.exists(f"{vectorstore_path}_docs.pkl"):
        try:
            index = faiss.read_index(f"{vectorstore_path}.index")
            with open(f"{vectorstore_path}_docs.pkl", "rb") as f:
                documents = pickle.load(f)
            return index, documents
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None, None
    else:
        return None, None

# Load local LLM with HuggingFacePipeline
@st.cache_resource
def load_llm():
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
        return pipe
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# Load sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# Main app logic
def main():
    # Ensure vector store exists
    if not ensure_vectorstore():
        st.stop()
    
    # Load components
    index, documents = load_vectorstore()
    llm = load_llm()
    embedding_model = load_embedding_model()
    
    if index is None or documents is None:
        st.error("❌ Failed to load vector store. Please check the logs.")
        st.stop()
    
    if llm is None:
        st.error("❌ Failed to load language model. Please check the logs.")
        st.stop()
    
    if embedding_model is None:
        st.error("❌ Failed to load embedding model. Please check the logs.")
        st.stop()
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Ask Your Question")
    
    # Chat input
    query = st.text_input("Enter your question here 👇", placeholder="e.g., What is the electricity access in rural areas in Africa?")
    
    if query:
        with st.spinner("🤔 Thinking..."):
            try:
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
                
                # Display results
                st.success("🎯 **Answer:**")
                st.write(result)
                
                # Show relevant context
                with st.expander("📚 **Relevant Data Used**"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**{i+1}.** {doc['content']}")
                        st.markdown(f"   *Country: {doc['metadata']['country']}, Indicator: {doc['metadata']['indicator']}*")
                        st.markdown("---")
                
                # Show search scores
                with st.expander("🔍 **Search Relevance Scores**"):
                    for i, (idx, score) in enumerate(zip(I[0], D[0])):
                        st.write(f"Result {i+1}: {score:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.error("Please try a different question or check the logs.")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("## 📊 **About This Chatbot**")
        st.markdown("""
        This RAG-powered chatbot provides intelligent answers about Sustainable Development Goals (SDGs) using:
        
        - **Data Source**: World Bank SDG Indicators
        - **AI Model**: Google's flan-t5-base
        - **Embeddings**: Sentence Transformers
        - **Vector Search**: FAISS
        
        **Sample Questions:**
        - Electricity access in different regions
        - Clean fuel adoption trends
        - Education statistics
        - Employment rates
        """)
        
        st.markdown("---")
        st.markdown("## 🛠️ **Technical Info**")
        st.markdown(f"""
        - **Vector Store**: {len(documents) if documents else 0} documents
        - **Model**: flan-t5-base
        - **Embeddings**: all-MiniLM-L6-v2
        """)

# Run the main app
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("🛠️ Built using Sentence Transformers, FAISS, HuggingFace Transformers, and Streamlit")
st.markdown("🚀 Deployed on Render")
