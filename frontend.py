import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Document Chat")
st.write("Ask questions about your uploaded documents.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"query": query})

                if response.status_code == 200:
                    data = response.json()
                    st.success("Response:")
                    st.write(data.get("response"))
                else:
                    st.error("Backend returned an error.")
            except:
                st.error("Cannot connect to backend. Is FastAPI running?")