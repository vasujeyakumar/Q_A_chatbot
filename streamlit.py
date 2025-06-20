import streamlit as st
import os
import asyncio
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("API"))

# Streamed model call
async def get_response_stream(user_input):
    # This returns a generator when stream=True
    return await asyncio.to_thread(
        client.chat.completions.create,
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond clearly in English."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.5,
        max_tokens=300,
        stream=True
    )

def run_streaming(user_input):
    response_chunks = asyncio.run(get_response_stream(user_input))
    final_response = ""
    response_placeholder = st.empty()  # This will be updated in-place

    for chunk in response_chunks:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                final_response += delta.content
                response_placeholder.markdown(f"<div class='response-box'>{final_response}▌</div>", unsafe_allow_html=True)

    response_placeholder.markdown(f"<div class='response-box'>{final_response}</div>", unsafe_allow_html=True)

# Streamlit UI setup
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""<style>
html, body, [class*="css"] {
    font-size: 22px !important;
    font-family: 'Segoe UI', sans-serif;
}
.stTextInput > div > div > input {
    font-size: 22px !important;
    padding: 16px;
    border-radius: 10px;
}
.stButton > button {
    font-size: 20px !important;
    padding: 12px 24px;
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
    border: none;
}
.stMarkdown h1 {
    font-size: 42px;
    text-align: center;
    color: #3f51b5;
    margin-bottom: 0;
}
.stMarkdown h3 {
    font-size: 30px;
    margin-top: 40px;
    color: #ab47bc;
}
.response-box {
    background-color: #f0f4ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 22px;
    color: #333;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
}
</style>""", unsafe_allow_html=True)

# Layout
st.markdown("# 🤖 Groq LLaMA3 Chatbot")
st.markdown("Welcome to your personal AI assistant powered by **LLaMA 3 + Groq**. Ask anything below:")

# Input
user_input = st.text_input("💬 What would you like to ask?")

# Trigger response
if user_input:
    with st.spinner("Thinking..."):
        run_streaming(user_input)
