# app.py
import streamlit as st
import requests
import json
from typing import Optional

st.set_page_config(page_title="Connection Prep Agent (Free)", page_icon="🤝", layout="wide")
st.title("🤝 Connection Prep Agent — Free (Hugging Face)")

st.markdown(
    "Paste a LinkedIn profile (About/Experience) and optionally 1–3 recent posts. "
    "This app uses free Hugging Face models. **Do not commit API tokens** to GitHub."
)

# ---------------------------
# Helper to call Hugging Face Inference API
# ---------------------------
def hf_call(model_id: str, inputs: str, hf_token: str, params: Optional[dict] = None):
    if not hf_token:
        raise ValueError("Hugging Face token not found. Set HF_TOKEN in Streamlit secrets.")
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": inputs}
    if params:
        payload["parameters"] = params
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        # return the error body for debugging
        raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")
    return resp.json()

# ---------------------------
# UI Inputs
# ---------------------------
with st.expander("How to use (quick)"):
    st.write(
        "- Copy-paste the public LinkedIn text you can view (About / Experience / Headline)."
        "- Optionally paste 1–3 recent post texts."
        "- Add a short meeting goal (e.g., 'Intro call about product partnership')."
        "- Click Generate Brief. The app calls free HF models (you must add a HF token in Streamlit secrets)."
    )

col1, col2 = st.columns([2,1])
with col1:
    profile_text = st.text_area("LinkedIn Profile (About / Experience / Headline)", height=200,
                                placeholder="Paste the person's About / headline / experience here.")
    posts_text = st.text_area("Recent posts (optional) — 1–3 posts (short)", height=160,
                              placeholder="Paste 1–3 short recent posts (optional).")
    meeting_goal = st.text_input("Meeting goal (optional)", placeholder="E.g., 'Intro: explore partnership'")

with col2:
    st.header("Settings")
    summarizer_model = st.selectbox("Summarizer model (smaller = faster):",
                                    ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"])
    generator_model = st.selectbox("Generator model (instruction):",
                                   ["google/flan-t5-small", "google/flan-t5-base"])
    bullets = st.slider("Max bullets per list", 3, 8, 5)
    st.caption("If the app fails due to resource limits, choose the smaller models and shorter inputs.")

# ---------------------------
# Get HF token from Streamlit secrets (secure)
# ---------------------------
hf_token = None
try:
    # Streamlit secrets: Add a secret named HF_TOKEN in Streamlit app settings
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = None

# ---------------------------
# Generate Brief logic
# ---------------------------
def make_prompt(profile_summary: str, posts_summary: str, goal: str, bullets: int):
    prompt = f"""
You are a concise assistant creating a one-page connection/meeting brief for a professional networking call.
Use the inputs below to produce clearly labeled sections. Keep language neutral, factual, and useful.

PROFILE_SUMMARY:
{profile_summary}

POSTS_SUMMARY:
{posts_summary}

MEETING_GOAL:
{goal}

Produce the output with these labeled sections and bullet lists:

PROFILE SNAPSHOT:
- 3 short lines about seniority, domain, value-focus.

TOP SKILLS:
- Up to {bullets} short bullets (skill/tool names).

TALKING POINTS:
- Up to {bullets} actionable topics based on posts/profile.

ICEBREAKERS:
- 3 friendly openers tailored to the person.

OPENING_QUESTION:
- One smart open-ended question.

SAMPLE_DM:
- A 2–3 sentence LinkedIn DM to request a short chat.

MEETING_AGENDA:
- 3 bullets for a 15–30 minute call.

If information is missing, be conservative and generic. Do not invent companies or facts.
"""
    return prompt.strip()

def generate_brief(profile_text, posts_text, goal, hf_token):
    # Short-circuit if no token
    if not hf_token:
        return {"error": "No Hugging Face token found. Add HF_TOKEN to Streamlit secrets (see app instructions)."}

    # 1) Summarize profile and posts using the summarizer model
    profile_summary = ""
    posts_summary = ""
    try:
        if profile_text.strip():
            out = hf_call(summarizer_model, profile_text, hf_token,
                          params={"max_new_tokens": 150})
            # HF summarizers return [{'summary_text': "..."}]
            if isinstance(out, list) and "summary_text" in out[0]:
                profile_summary = out[0]["summary_text"]
            elif isinstance(out, dict) and "summary_text" in out:
                profile_summary = out["summary_text"]
            else:
                profile_summary = profile_text[:400]
        if posts_text.strip():
            out2 = hf_call(summarizer_model, posts_text, hf_token, params={"max_new_tokens": 120})
            if isinstance(out2, list) and "summary_text" in out2[0]:
                posts_summary = out2[0]["summary_text"]
            else:
                posts_summary = posts_text[:300]
    except Exception as e:
        return {"error": f"Summarization failed: {e}"}

    # 2) Use generator model to create the structured brief
    prompt = make_prompt(profile_summary, posts_summary, goal, bullets)
    try:
        gen_out = hf_call(generator_model, prompt, hf_token,
                          params={"max_new_tokens": 400, "temperature": 0.2})
        # HF text2text usually returns [{"generated_text": "..."}] or plain text
        if isinstance(gen_out, list) and "generated_text" in gen_out[0]:
            text = gen_out[0]["generated_text"]
        elif isinstance(gen_out, dict) and "generated_text" in gen_out:
            text = gen_out["generated_text"]
        else:
            # some models return string directly
            text = json.dumps(gen_out)
    except Exception as e:
        return {"error": f"Generation failed: {e}"}

    return {"brief": text, "profile_summary": profile_summary, "posts_summary": posts_summary}

# Run generation on button click
if st.button("🚀 Generate Brief"):
    with st.spinner("Working..."):
        result = generate_brief(profile_text, posts_text, meeting_goal, hf_token)

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("📄 Connection Prep Brief")
        st.markdown(result["brief"])
        st.download_button("⬇️ Download as Markdown", result["brief"], file_name="connection_prep_brief.md")
