# src/streamlit_app.py

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Vietnamese Medical Chatbot Demo",
    page_icon="🩺",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #007bff;
        color: white;
    }
    .medical-disclaimer {
        font-size: 0.8em;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding-top: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model_choice = st.selectbox("Choose Model", ["gpt-4o-mini"])
    st.info("This is a demo for the **TVAFT** Vietnamese Medical QA project.")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
st.title("🩺 Vietnamese Medical Chatbot")
st.markdown("Cố vấn sức khỏe thông minh cho người Việt.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Logic ---
if prompt := st.chat_input("Hỏi tôi về vấn đề sức khỏe của bạn..."):
    # Check for API Key
    if not api_key:
        st.error("Please provide an OpenAI API Key in the sidebar.")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare OpenAI client
    client = OpenAI(api_key=api_key)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # System prompt to simulate the medical QA behavior
        system_prompt = (
            "Bạn là một trợ lý y tế thông minh dành cho người Việt Nam. "
            "Hãy trả lời các câu hỏi y tế một cách chính xác, chuyên nghiệp và ân cần. "
            "Sử dụng thuật ngữ y khoa chính xác khi cần thiết. "
            "Luôn nhắc nhở người dùng rằng thông tin này chỉ mang tính tham khảo và nên đi khám bác sĩ."
        )

        try:
            for response in client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")

# --- Footer/Disclaimer ---
st.markdown("""
<div class="medical-disclaimer">
    <strong>Tuyên bố miễn trừ trách nhiệm:</strong> Đây là một hệ thống thử nghiệm sử dụng trí tuệ nhân tạo. 
    Các câu trả lời chỉ mang tính chất tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên môn cho các vấn đề sức khỏe nghiêm trọng.
</div>
""", unsafe_allow_html=True)
