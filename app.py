import streamlit as st
import requests
import os

st.set_page_config(page_title="Connection Prep Agent", page_icon="🤝")

st.title("🤝 Connection Prep Agent")

st.write("Paste LinkedIn posts + meeting goal, and I’ll draft a connection brief for you.")

# Input fields
posts = st.text_area("Paste 1–3 recent LinkedIn posts from the person:", height=150)
goal = st.text_area("What’s your meeting goal?", height=100)

# Button
if st.button("Generate Brief"):
    if not posts.strip() or not goal.strip():
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Generating brief..."):

            # Hugging Face Inference API
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}

            payload = {
                "inputs": f"Summarize this person's interests and align with this goal: {goal}\n\nPosts:\n{posts}",
                "parameters": {"max_length": 200, "temperature": 0.7}
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                try:
                    data = response.json()
                    summary = data[0]['summary_text'] if isinstance(data, list) else data
                    st.subheader("Brief (Unstructured)")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Unexpected response: {response.text}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
