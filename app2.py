import streamlit as st
import pandas as pd
import numpy as np
import functools
from langchain_core.messages import AIMessage
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import os
from kk import chatbot

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# CSS for minimalistic styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2em;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .chat-container {
        border-radius: 10px;
        padding: 15px;
        background-color: #f5f5f5;
        margin: 10px 0;
    }
    .user-chat, .bot-chat {
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 8px;
    }
    .user-chat {
        background-color: #e1f5fe;
        text-align: right;
    }
    .bot-chat {
        background-color: #ffebee;
        text-align: left;
    }
    .input-box {
        margin-top: 20px;
    }
    .send-button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.markdown('<div class="main-title">Minimalist Chat Interface</div>', unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for message
user_message = st.text_input("You: ", "", placeholder="Type your message here...", key="input", label_visibility="collapsed")

# Send button
if st.button("Send", key="send", help="Send your message"):
    if user_message:
        # Append user's message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_message})
        # Simulate bot response
        llm = ChatOpenAI(model="gpt-4o")
        bot_response = chatbot(user_message)
        response = bot_response['messages'][-1].content
        st.session_state.chat_history.append({"role": "bot", "message": response})

# Display chat history with improved layout
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f'<div class="chat-container user-chat"><strong>You:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container bot-chat"><strong>Bot:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
