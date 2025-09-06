"""
Core chatbot functionality for Islamic History Chatbot
"""
import os
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from .vectorstore import load_or_create_vectorstore
from .utils import validate_environment


def create_qa_chain(vectorstore):
    """
    Create the question-answering chain with optimized retrieval and prompt.
    
    Args:
        vectorstore: The FAISS vectorstore containing the document embeddings
        
    Returns:
        The configured retrieval chain
    """
   # load_dotenv()
    # Configure retriever with MMR for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 6,  # Number of documents to retrieve
            "fetch_k": 12,  # Number of candidates to consider
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )

    #load_dotenv()  # This loads the .env file
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        request_timeout=60
    )

    # Create the system prompt
    system_prompt = """
You are an expert Islamic historian and scholar specializing in the life of Prophet Muhammad (peace be upon him). 
Your knowledge is based exclusively on Martin Lings' biography "Muhammad: His Life Based on the Earliest Sources."

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context below
2. When asked for timelines or chronological events, organize information in proper chronological order
3. For major life events, provide comprehensive details including dates, locations, and circumstances when available
4. Always include page references from the source material when possible
5. If the context contains relevant but incomplete information, synthesize what you can and specify what might be missing
6. Be thorough and detailed when sufficient context is provided
7. Maintain a respectful and scholarly tone appropriate for Islamic history
8. If you cannot find sufficient information in the context, clearly state: "Based on the provided context from Martin Lings' biography, I cannot find sufficient information to fully answer this question."

Remember: You are answering questions about the most beloved figure in Islamic history. Maintain accuracy, respect, and scholarly precision.

Context from Martin Lings' biography:
{context}

Question: {input}

Provide a comprehensive and respectful answer based on the context above:"""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


@st.cache_resource(show_spinner=False)
def initialize_chatbot() -> Any:
    """
    Initialize the chatbot by loading the vectorstore and creating the QA chain.
    Uses Streamlit caching to avoid reinitializing on every interaction.
    
    Returns:
        Configured QA chain ready for questions
    """
    try:
        # Validate environment variables
        validate_environment()
        
        # Define PDF path
        pdf_path = "docs/muhammad_martin_Lings.pdf"
        
        # Check if PDF exists
        # if not os.path.exists(pdf_path):
            # raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please ensure the file is uploaded.")
        
        # Load or create vectorstore
        vectorstore = load_or_create_vectorstore(pdf_path)
        
        # Create QA chain
        qa_chain = create_qa_chain(vectorstore)
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        raise e


def get_response(qa_chain, question: str) -> Dict[str, Any]:
    """
    Get response from the chatbot for a given question.
    
    Args:
        qa_chain: The initialized QA chain
        question: User's question
        
    Returns:
        Dictionary containing the answer and sources
    """
    try:
        # Get response from the chain
        result = qa_chain.invoke({"input": question.strip()})
        
        # Extract answer
        answer = result.get("answer", "I couldn't generate a response.")
        
        # Extract sources (page numbers)
        sources = []
        if 'context' in result and result['context']:
            sources = [
                doc.metadata.get("page", "unknown") 
                for doc in result["context"]
                if doc.metadata.get("page") is not None
            ]
            # Remove duplicates while preserving order
            sources = list(dict.fromkeys(sources))
        
        return {
            "answer": answer,
            "sources": sources,
            "context_docs": result.get("context", [])
        }
        
    except Exception as e:
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
            "sources": [],
            "context_docs": []
        }


def debug_retrieval(qa_chain, question: str, show_content: bool = False) -> None:
    """
    Debug function to show what documents are being retrieved for a question.
    
    Args:
        qa_chain: The initialized QA chain
        question: The question to debug
        show_content: Whether to show document content preview
    """
    try:
        # Get the retriever from the chain
        retriever = qa_chain.retriever
        
        # Retrieve documents
        docs = retriever.invoke(question)
        
        st.write(f"🔍 **Debug Info for:** '{question}'")
        st.write(f"**Retrieved {len(docs)} documents:**")
        
        for i, doc in enumerate(docs):
            page = doc.metadata.get('page', 'unknown')
            st.write(f"**Doc {i+1}:** Page {page}")
            
            if show_content:
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.write(f"*Content:* {content_preview}")
            
            st.write("---")
            
    except Exception as e:
        st.error(f"Debug failed: {str(e)}")