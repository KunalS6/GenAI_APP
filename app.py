import os
from dotenv import load_dotenv
import streamlit as st

# =======================
# ENV
# =======================
load_dotenv()
assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY is missing"

from langchain_groq import ChatGroq
#from langchain.globals import set_verbose
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#set_verbose(False)

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="LLM Chat App",
    page_icon="🤖",
    layout="wide"
)

# =======================
# CUSTOM CSS
# =======================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.chat-container {
    max-width: 900px;
    margin: auto;
}

.user-msg {
    background: #2563eb;
    color: white;
    padding: 12px 16px;
    border-radius: 16px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    margin-left: auto;
}

.assistant-msg {
    background: #1e293b;
    color: #e5e7eb;
    padding: 12px 16px;
    border-radius: 16px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
}

.header {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("**Model:** llama-3.1-8b-instant")
    st.markdown("**Provider:** Groq")
    st.divider()

    if st.button("🧹 New Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("Built with ❤️ using LangChain & Streamlit")

# =======================
# HEADER
# =======================
st.markdown("""
<div class="header">
    <h1>🤖 LLM Chat Assistant</h1>
    <p>Fast • Free • Powered by Groq</p>
</div>
""", unsafe_allow_html=True)

# =======================
# SESSION STATE
# =======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =======================
# MODEL & CHAIN
# =======================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise assistant."),
        ("user", "{question}")
    ]
)

chain = prompt | llm | StrOutputParser()

# =======================
# CHAT UI
# =======================
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-msg'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='assistant-msg'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# =======================
# INPUT
# =======================
user_input = st.chat_input("Type your message…")

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()

# =======================
# FOOTER
# =======================
st.markdown("""
<div class="footer">
    © 2026 • LLM Chat App • Production Ready
</div>
""", unsafe_allow_html=True)
