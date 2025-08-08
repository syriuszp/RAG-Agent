import os
import datetime
from dotenv import load_dotenv
import streamlit as st

from gui import setup_page, sidebar, display_conversation, clear_button, question_form
from data_sources import source_manager
from processing import compute_embeddings, answer_question

# Load environment variables
load_dotenv()

# Setup page
setup_page()

# API Keys from environment variables
cohere_api = os.getenv("COHERE_API_KEY")
if not cohere_api:
    st.warning("Missing COHERE_API_KEY in environment")

gemini_api = os.getenv("GEMINI_API_KEY")
if not gemini_api:
    st.warning("Missing GEMINI_API_KEY in environment")

# Initialize session state
if "items" not in st.session_state:
    st.session_state["items"] = []
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Sidebar - manage content
uploaded_files = sidebar(st.session_state["items"])
if uploaded_files:
    new_items = source_manager.load(uploaded_files)
    existing_names = set((item["name"], item["type"]) for item in st.session_state["items"])
    added = [item for item in new_items if (item["name"], item["type"]) not in existing_names]
    st.session_state["items"].extend(added)
    st.success(f"Uploaded {len(uploaded_files)} file(s) with {len(added)} total items.")

# Main interface - chat
st.subheader("Chat with Your Visual Data")

display_conversation(st.session_state["conversation"])
clear_button(st.session_state["conversation"])

question, submit_button = question_form()

if submit_button:
    if not cohere_api or not gemini_api:
        st.error("Please provide both Cohere and Gemini API keys.")
    elif not question:
        st.error("Please enter a question.")
    elif not st.session_state["items"]:
        st.error("No content loaded to search.")
    else:
        with st.spinner("Computing embeddings and searching..."):
            compute_embeddings(st.session_state["items"], cohere_api)
            answer, best_item, sim = answer_question(
                question,
                st.session_state["items"],
                cohere_api,
                gemini_api,
            )
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state["conversation"].append(
                {
                    "question": question,
                    "answer": answer,
                    "relevant_image": best_item["name"],
                    "timestamp": timestamp,
                    "image_display": best_item["img"],
                    "similarity": sim,
                }
            )
            st.rerun()
