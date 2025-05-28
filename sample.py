
import streamlit as st
from transformers import pipeline

# Load pre-trained summarizer pipeline
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()

# Streamlit UI
st.title("📝 Text Summarizer App")

text = st.text_area("Paste your text below for summarization:", height=300)

if st.button("Summarize"):
    if text:
        with st.spinner("Generating summary..."):
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
            st.subheader("Summary:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text!")

st.markdown("---")
st.caption("Built using Hugging Face & Streamlit 🚀")