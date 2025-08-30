import streamlit as st
from transformers import pipeline

# Load summarizer (free, runs via CPU in Streamlit Cloud)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

def summarize_text(text: str) -> str:
    """Summarize text safely by chunking long inputs."""
    if not text.strip():
        return "⚠️ Please provide some text to summarize."

    max_chunk = 400
    sentences = text.split(". ")
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_new_tokens=120,   # only using max_new_tokens
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Error on chunk: {e}]")

    return " ".join(summaries)


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Connection Prep Agent", layout="centered")

st.title("🤝 Connection Prep Agent")
st.write("Paste any LinkedIn post, article, or text. I’ll summarize it for quick prep before outreach.")

user_input = st.text_area("✍️ Paste text to summarize:", height=200)

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        summary = summarize_text(user_input)
        st.subheader("📌 Summary")
        st.write(summary)
