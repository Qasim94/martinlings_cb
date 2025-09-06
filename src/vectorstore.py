"""
Vector store operations for the Islamic History Chatbot
"""

import os
import streamlit as st
from typing import Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


@st.cache_resource(show_spinner=False)
def load_or_create_vectorstore(pdf_path: str) -> Any:
    """
    Load existing FAISS index or create new one from PDF.
    Uses Streamlit caching to avoid rebuilding the index every time.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        FAISS vectorstore instance
    """
    index_path = "data/faiss_index"
    
    # Try to load existing index

   # load_dotenv()  # This loads the .env file
    if os.path.exists(index_path):
        try:
            st.info("📚 Loading existing knowledge base...")
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("✅ Knowledge base loaded successfully!")
            return vectorstore
            
        except Exception as e:
            st.warning(f"⚠️ Could not load existing index: {str(e)}")
            st.info("🔄 Building new knowledge base...")
    
    # Create new vectorstore
    vectorstore = build_vectorstore(pdf_path)
    return vectorstore


def build_vectorstore(pdf_path: str) -> Any:
    """
    Build a new FAISS vectorstore from the PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        FAISS vectorstore instance
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        load_dotenv()  # This loads the .env file
        # Load PDF
        st.info("📖 Loading PDF document...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        st.success(f"✅ Loaded {len(docs)} pages from PDF")

        # Split documents into chunks
        st.info("✂️ Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
        st.success(f"✅ Created {len(chunks)} text chunks")

        # Create embeddings
        st.info("🧠 Creating embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

        # Build FAISS index
        st.info("🔍 Building search index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save the index
        index_path = "data/faiss_index"
        vectorstore.save_local(index_path)
        st.success("💾 Knowledge base saved successfully!")

        # Calculate and display index size
        try:
            index_size = sum(
                os.path.getsize(os.path.join(index_path, f)) 
                for f in os.listdir(index_path)
            )
            st.info(f"📊 Index size: {index_size / (1024*1024):.1f} MB")
        except:
            pass

        return vectorstore

    except Exception as e:
        st.error(f"❌ Error building vectorstore: {str(e)}")
        raise e


def get_vectorstore_info(vectorstore) -> dict:
    """
    Get information about the vectorstore.
    
    Args:
        vectorstore: The FAISS vectorstore
        
    Returns:
        Dictionary with vectorstore information
    """
    try:
        # Get number of vectors
        index_size = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else "Unknown"
        
        # Get embedding dimension
        dimension = vectorstore.index.d if hasattr(vectorstore, 'index') else "Unknown"
        
        return {
            "num_vectors": index_size,
            "embedding_dimension": dimension,
            "index_type": "FAISS"
        }
    except:
        return {
            "num_vectors": "Unknown",
            "embedding_dimension": "Unknown", 
            "index_type": "FAISS"
        }


def search_similar_chunks(vectorstore, query: str, k: int = 5) -> list:
    """
    Search for similar chunks in the vectorstore.
    
    Args:
        vectorstore: The FAISS vectorstore
        query: Search query
        k: Number of results to return
        
    Returns:
        List of similar documents
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def add_documents_to_vectorstore(vectorstore, new_docs: list, save_path: str = "data/faiss_index") -> Any:
    """
    Add new documents to existing vectorstore.
    
    Args:
        vectorstore: Existing FAISS vectorstore
        new_docs: List of new documents to add
        save_path: Path to save updated index
        
    Returns:
        Updated vectorstore
    """
    try:
        # Add documents
        vectorstore.add_documents(new_docs)
        
        # Save updated index
        vectorstore.save_local(save_path)
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to add documents: {str(e)}")
        return vectorstore