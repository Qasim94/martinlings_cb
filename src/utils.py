"""
Utility functions for the Islamic History Chatbot
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List


def load_sample_questions() -> List[str]:
    """
    Load sample questions for the sidebar.
    
    Returns:
        List of sample questions
    """
    return [
        "Major events in Prophet Muhammad's life before Hijra in chronological order?",
        "Tell me about the Prophet's birth and early childhood",
        "What happened during the first revelation in the Cave of Hira?",
        "Who was Khadijah and what role did she play in the Prophet's life?",
        "Describe the persecution of early Muslims in Mecca",
        "What was the Year of Sorrow (Am al-Huzn)?",
        "Tell me about the Prophet's journey to Ta'if",
        "What was the Night Journey (Isra and Mi'raj)?",
        "Who was Abu Bakr and how did he support the Prophet?",
        "Describe the Pledge of Aqaba",
        "What led to the decision to migrate to Medina?",
        "How did the Prophet's character earn him the title 'Al-Amin'?",
        "Tell me about the Prophet's relationship with his uncle Abu Talib",
        "What was the boycott of Banu Hashim?",
        "Describe the reconstruction of the Ka'aba"
    ]


def format_sources(sources: List[str]) -> str:
    """
    Format source page numbers for display.
    
    Args:
        sources: List of page numbers
        
    Returns:
        Formatted string with sources
    """
    if not sources:
        return "**Sources:** No specific pages referenced"
    
    # Remove duplicates and sort
    unique_sources = sorted(list(set(str(s) for s in sources if s != "unknown")))
    
    if not unique_sources:
        return "**Sources:** Page numbers not available"
    
    if len(unique_sources) == 1:
        return f"**Source:** Page {unique_sources[0]}"
    elif len(unique_sources) <= 5:
        return f"**Sources:** Pages {', '.join(unique_sources)}"
    else:
        # Show first few and indicate there are more
        return f"**Sources:** Pages {', '.join(unique_sources[:5])} and {len(unique_sources)-5} more"


def validate_environment() -> None:
    """
    Validate that required environment variables are set.
    
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please set them in your .env file or Streamlit secrets."
        )


def load_css(file_path: str) -> None:
    """
    Load custom CSS file.
    
    Args:
        file_path: Path to CSS file
    """
    try:
        if os.path.exists(file_path):
            with open(file_path) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        # Silently fail if CSS can't be loaded
        pass


def display_chat_stats(messages: List[dict]) -> None:
    """
    Display chat statistics in the sidebar.
    
    Args:
        messages: List of chat messages
    """
    if not messages:
        return
    
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    
    st.sidebar.markdown("### 📊 Chat Stats")
    st.sidebar.metric("Questions Asked", len(user_messages))
    st.sidebar.metric("Responses Given", len(assistant_messages))


def export_chat_history(messages: List[dict]) -> str:
    """
    Export chat history as formatted text.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted chat history as string
    """
    if not messages:
        return "No chat history available."
    
    export_text = "# Islamic History Chatbot - Chat History\n\n"
    
    for i, message in enumerate(messages):
        role = "🤖 Assistant" if message["role"] == "assistant" else "👤 User"
        export_text += f"## {role}\n\n{message['content']}\n\n---\n\n"
    
    return export_text


def get_question_suggestions(current_question: str) -> List[str]:
    """
    Get related question suggestions based on current question.
    
    Args:
        current_question: The current question being asked
        
    Returns:
        List of related questions
    """
    # Simple keyword-based suggestions
    suggestions_map = {
        "birth": [
            "Tell me about the Year of the Elephant",
            "Who was Abdul Muttalib?",
            "Describe the Prophet's early childhood"
        ],
        "khadijah": [
            "How did the Prophet become a merchant?",
            "What was Khadijah's role in early Islam?",
            "Tell me about the Prophet's marriage"
        ],
        "revelation": [
            "What was the Prophet's reaction to first revelation?",
            "Who was Waraqah ibn Nawfal?",
            "How did early Muslims respond to the message?"
        ],
        "persecution": [
            "Tell me about the migration to Abyssinia",
            "What was the boycott of Banu Hashim?",
            "How did the Prophet deal with opposition?"
        ]
    }
    
    current_lower = current_question.lower()
    for keyword, suggestions in suggestions_map.items():
        if keyword in current_lower:
            return suggestions[:3]  # Return up to 3 suggestions
    
    return []


def clean_text(text: str) -> str:
    """
    Clean and format text for better display.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Ensure proper sentence spacing
    text = text.replace('. ', '. ').replace('..', '.')
    
    return text.strip()


def get_app_info() -> dict:
    """
    Get information about the app.
    
    Returns:
        Dictionary with app information
    """
    return {
        "name": "Islamic History Chatbot",
        "version": "1.0.0",
        "description": "AI-powered chatbot for exploring Islamic history",
        "source": "Martin Lings' Biography of Prophet Muhammad (PBUH)",
        "technology": "Streamlit + OpenAI + LangChain"
    }


def log_interaction(question: str, response: str, sources: List[str]) -> None:
    """
    Log user interactions for analytics (optional).
    
    Args:
        question: User's question
        response: Bot's response
        sources: Source pages referenced
    """
    # This could be extended to log to a file or analytics service
    # For now, it's a placeholder for future functionality
    pass