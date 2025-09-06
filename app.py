import os
import streamlit as st
from src.chatbot import initialize_chatbot, get_response
from src.utils import load_sample_questions, format_sources, load_css

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Islamic History Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css('static/css/style.css')

# -------------------------------
# Title and description
# -------------------------------
st.title("📚 Islamic History Chatbot")
st.markdown("### Ask questions about the life of Prophet Muhammad (PBUH)")
st.markdown("*Based on Martin Lings' acclaimed biography*")

# -------------------------------
# Initialize chatbot
# -------------------------------
if "chatbot_initialized" not in st.session_state:
    with st.spinner("🔄 Initializing chatbot and loading knowledge base..."):
        try:
            st.session_state.qa_chain = initialize_chatbot()
            st.session_state.chatbot_initialized = True
            st.success("✅ Chatbot is ready!")
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {str(e)}")
            st.stop()

# -------------------------------
# Initialize chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Assalamu Alaikum! I'm here to help you learn about the life of Prophet Muhammad (peace be upon him) based on Martin Lings' biography. What would you like to know?"
        }
    ]

# Pending question mechanism
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -------------------------------
# Sidebar: Sample questions
# -------------------------------
with st.sidebar:
    st.header("📝 Sample Questions")
    st.markdown("Click on any question to ask it:")

    sample_questions = load_sample_questions()

    for i, question in enumerate(sample_questions):
        if st.button(question, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = question

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask for chronological timelines
    - Request specific page references
    - Inquire about early life events
    - Ask about relationships and family
    - Request details about revelations
    """)

# -------------------------------
# Handle pending sidebar question
# -------------------------------
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    try:
        result = get_response(st.session_state.qa_chain, prompt)
        response_text = result["answer"]
        if result.get("sources"):
            response_text += f"\n\n{format_sources(result['sources'])}"

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# -------------------------------
# Display chat messages
# -------------------------------
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# -------------------------------
# Chat input
# -------------------------------
if prompt := st.chat_input("Ask about the Prophet's life..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching through the biography..."):
            try:
                result = get_response(st.session_state.qa_chain, prompt)
                response_text = result["answer"]
                if result.get("sources"):
                    response_text += f"\n\n{format_sources(result['sources'])}"

                # Display and store
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            Built with ❤️ using Streamlit and OpenAI • 
            Based on "Muhammad: His Life Based on the Earliest Sources" by Martin Lings
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
