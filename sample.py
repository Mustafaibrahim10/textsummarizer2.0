import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 📥 Ensure NLTK 'punkt' tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ✅ Streamlit page configuration
st.set_page_config(page_title="Text Summarizer", page_icon="🧠", layout="wide")

# ✅ Load summarization pipeline
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()

# 🔹 Abstractive summarization using Hugging Face pipeline
def abstractive_summary(text):
    summary = summarizer(text, max_length=200, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# 🔹 Extractive summarization using TF-IDF & Cosine Similarity
def extractive_summary(text, num_sentences=3):
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = text.split('. ')  # fallback
    if len(sentences) <= num_sentences:
        return text
    tfidf = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf, tfidf)
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[-num_sentences:]]
    return ' '.join(ranked_sentences)

# 🎯 Sidebar info
with st.sidebar:
    st.title("📋 App Info")
    st.markdown("""
    **🧠 Text Summarizer App**

    - Summarize long articles  
    - Extractive or Abstractive  
    - NLP + Transformers

    ---
    **👨‍💻 Developer**  
    *Mustafa Ibrahim*  
    **📧 Contact**: iammustafaibrahim1012@gmail.com  
    """)
    st.markdown("---")
    st.markdown("✨ _Thank you for using the app!_")

# 🧠 Main App UI
st.markdown("<h1 style='text-align: center;'>🧠 Text Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Summarize your content using cutting-edge NLP techniques and Transformers.</p>", unsafe_allow_html=True)

# 📝 User Input
text_input = st.text_area("📌 Paste your text below:", height=300, placeholder="Enter news article, blog, or essay...")

# 🎛️ Options
col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("🧪 Choose Summarization Mode:", ["Extractive", "Abstractive"])
with col2:
    num_sentences = st.slider("📏 Extractive: No. of sentences", 1, 10, 3)

# 🚀 Summarize Button
if st.button("🧠 Generate Summary"):
    if not text_input.strip():
        st.warning("Please enter text before summarizing.")
    else:
        with st.spinner("🔍 Summarizing..."):
            if mode == "Extractive":
                summary = extractive_summary(text_input, num_sentences)
            else:
                summary = abstractive_summary(text_input)
        st.subheader("📄 Summary Output")
        st.success(summary)

# 👣 Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray; font-size: 0.9em;'>
        Made with ❤️ using <a href='https://streamlit.io/' target='_blank'>Streamlit</a> by <strong>Mustafa Ibrahim</strong>
    </p>
""", unsafe_allow_html=True)
