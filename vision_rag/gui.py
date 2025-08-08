"""Streamlit GUI components for VisionRAG."""

import streamlit as st
from PIL import Image


def setup_page() -> None:
    st.set_page_config(page_title="VisionRAG: Multimodal Search & VQA", layout="wide")
    st.title("VisionRAG: Multimodal Search & Visual Question Answering")


def sidebar(items):
    st.sidebar.header("Content Sources")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Images or PDFs",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
        help="Upload images (PNG, JPG, JPEG) or PDF documents",
    )

    st.sidebar.header("\U0001F4C1 Loaded Content")
    if items:
        st.sidebar.markdown(f"**Total items:** {len(items)}")
        preview_items = items[:6]
        for item in preview_items:
            with st.sidebar.container():
                small_img = item["img"].resize((120, 120), Image.Resampling.LANCZOS)
                caption = item["name"][:15] + "..." if len(item["name"]) > 15 else item["name"]
                st.image(small_img, caption=caption, width=120)
        if len(items) > 6:
            st.sidebar.markdown(f"*... and {len(items) - 6} more items*")
    else:
        st.sidebar.info("No content loaded yet.")
    return uploaded_files


def display_conversation(conversation):
    if conversation:
        st.markdown("### Conversation History")
        for exchange in conversation:
            with st.expander(f"Q: {exchange['question']} ({exchange['timestamp']})", expanded=False):
                st.markdown(f"**Question:** {exchange['question']}")
                st.markdown(f"**Answer:** {exchange['answer']}")
                st.markdown(f"**Relevant Image:** {exchange['relevant_image']}")
                if exchange.get("image_display"):
                    st.image(exchange["image_display"], caption=exchange["relevant_image"], use_container_width=True)


def clear_button(conversation):
    if conversation and st.button("\U0001F5D1\uFE0F Clear Conversation"):
        st.session_state["conversation"] = []
        st.rerun()


def question_form():
    with st.form("question_form"):
        question = st.text_input("Ask a question about your visual data:")
        submit_button = st.form_submit_button("Send", type="primary")
    return question, submit_button
