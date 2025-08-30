import json
import textwrap
from typing import Dict, List
import streamlit as st

from transformers import pipeline

st.set_page_config(page_title="Connection Prep Agent", page_icon="🤝", layout="wide")

st.title("🤝 Connection Prep Agent")
st.caption("Paste a LinkedIn profile (and optionally their recent posts) to get a 1‑page meeting brief. "
           "All analysis runs locally in this app using open-source models. No scraping, no automation.")

with st.expander("How to use (and what's legal)", expanded=False):
    st.markdown("""
- **Copy-paste** the profile text you can already view on LinkedIn (About, Experience, Skills, Education).
- Optionally paste **their recent posts** (copy a few posts or highlights).
- Click **Generate Brief**. The app runs open-source language models to create a one‑pager.
- This is **legal** because you're only analyzing content you can access, with no bots or scraping.
- For privacy, don't paste confidential data.
    """)

# Sidebar: settings
st.sidebar.header("Settings")
engine = st.sidebar.selectbox(
    "Engine (choose smaller if you're on low memory)",
    [
        "Fast (FLAN-T5-small + DistilBART)",
        "Quality (FLAN-T5-base + DistilBART)",
        "Summarization‑only (fastest, least smart)",
    ],
    index=0
)

max_output_bullets = st.sidebar.slider("Max bullets per section", 3, 12, 6)
include_agenda = st.sidebar.checkbox("Include meeting agenda suggestions", value=True)
include_dm_email = st.sidebar.checkbox("Include sample DM + email subject", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Tips")
st.sidebar.markdown("""
- If you hit **out of memory** on free hosting, switch to **Summarization‑only** or **Fast** engine.
- Shorter input → faster & better output.
- Paste 1–3 recent posts, not 20.
""")

# Lazy-load models
@st.cache_resource(show_spinner=True)
def load_models(engine_name: str):
    if engine_name == "Summarization‑only (fastest, least smart)":
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        generator = None
    elif engine_name == "Quality (FLAN-T5-base + DistilBART)":
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
    else:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return summarizer, generator

def chunk_text(s: str, chunk_size: int = 1600, overlap: int = 150) -> List[str]:
    s = s.strip()
    if not s:
        return []
    chunks = []
    start = 0
    n = len(s)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = s[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def summarize_long_text(summarizer, text: str, max_len: int = 220, min_len: int = 80) -> str:
    if not text or not text.strip():
        return ""
    parts = chunk_text(text, 1600, 150)
    summaries = []
    for p in parts:
        out = summarizer(p, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(out[0]["summary_text"])
    if len(summaries) == 1:
        return summaries[0]
    # Merge summaries into a final summary
    merged = " ".join(summaries)
    final = summarizer(merged, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
    return final

def generate_brief(profile_text: str, posts_text: str, goal_text: str, engine_name: str,
                   bullets: int, include_agenda: bool, include_dm_email: bool) -> Dict:
    summarizer, generator = load_models(engine_name)

    # Compact versions of inputs
    profile_sum = summarize_long_text(summarizer, profile_text, 220, 80) if profile_text else ""
    posts_sum = summarize_long_text(summarizer, posts_text, 150, 60) if posts_text else ""
    goal_sum = goal_text.strip()

    if generator is None:
        # Summarization-only fallback: heuristic brief
        brief = {
            "profile_snapshot": profile_sum[:700],
            "top_skills": [],
            "domain_knowledge": [],
            "mutual_interests": [],
            "talking_points": [posts_sum] if posts_sum else [],
            "icebreakers": [],
            "opening_question": "",
            "sample_dm": "",
            "email_subject": "",
            "meeting_agenda": []
        }
        return brief

    # Instruction to the generator model
    template = f"""
You are an assistant that prepares a crisp, tactful 1‑page meeting brief based on a LinkedIn profile and (optional) recent posts.
Return **only valid JSON** with the following keys:

- profile_snapshot: 3 short lines summarizing the person's seniority, domain, and value focus.
- top_skills: up to {bullets} bullet phrases of key skills or tools (short, no sentences).
- domain_knowledge: up to {bullets} bullet phrases of industry/functional expertise.
- mutual_interests: infer 3–{max(3, min(bullets, 6))} items from education, locations, groups, or interests.
- talking_points: up to {bullets} specific, positive topics to discuss based on their posts/activities (if posts provided).
- icebreakers: 3–5 friendly openers tailored to the person.
- opening_question: one smart, respectful, open-ended question.
{"- sample_dm: 2–3 sentence LinkedIn DM to request a short chat." if include_dm_email else ""}
{"- email_subject: short subject line for an intro email." if include_dm_email else ""}
{"- meeting_agenda: 3–5 bullet agenda items for a 15–30 min call." if include_agenda else ""}

Write concise items. Be specific and avoid generic fluff.
If not enough info, make reasonable, neutral inferences. Do not hallucinate company names or facts not present.

INPUT:
[PROFILE]
{profile_text}

[PROFILE_SUMMARY]
{profile_sum}

[RECENT_POSTS_SUMMARY]
{posts_sum}

[MEETING_GOAL]
{goal_sum}
"""
    raw = generator(template, max_length=768, do_sample=False)[0]["generated_text"]

    # Try to parse JSON defensively
    def extract_json(s: str) -> Dict:
        s = s.strip()
        # Find the first { ... } block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        try:
            return json.loads(s)
        except Exception:
            # Last resort: lightweight fixes
            s = s.replace("\n- ", "\n")
            s = s.replace("—", "-").replace("–", "-")
            try:
                return json.loads(s)
            except Exception:
                return {"raw_text": s}

    return extract_json(raw)

# ---------- UI ----------

st.subheader("Paste Inputs")
profile_text = st.text_area("LinkedIn Profile Text (About, Experience, Skills, Education)", height=240,
                            placeholder="Copy-paste text from a LinkedIn profile you can view.")
posts_text = st.text_area("Recent Posts or Highlights (optional)", height=160,
                          placeholder="Paste 1–3 recent posts, or key highlights.")
goal_text = st.text_input("Meeting Context/Goal (optional)",
                          placeholder="e.g., 'First intro chat about AI in Customer Success'")

col_run, col_clear = st.columns([1,1])
with col_run:
    run = st.button("🚀 Generate Brief", type="primary")
with col_clear:
    if st.button("🧹 Clear All"):
        st.experimental_rerun()

if run:
    with st.spinner("Thinking..."):
        brief = generate_brief(profile_text, posts_text, goal_text, engine,
                               max_output_bullets, include_agenda, include_dm_email)

    st.success("Brief ready! Scroll down to view and download.")
    st.divider()

    # Render brief nicely
    def as_markdown(b: Dict) -> str:
        def join_bullets(key: str) -> str:
            arr = b.get(key, []) or []
            if isinstance(arr, str):
                arr = [arr]
            arr = [str(x).strip() for x in arr if str(x).strip()]
            return "".join([f"- {x}\n" for x in arr])

        md = f"""# Connection Prep Brief

**Snapshot**  
{b.get('profile_snapshot','').strip()}

## Top Skills
{join_bullets('top_skills')}

## Domain Knowledge
{join_bullets('domain_knowledge')}

## Mutual Interests
{join_bullets('mutual_interests')}

## Talking Points
{join_bullets('talking_points')}

## Icebreakers
{join_bullets('icebreakers')}

**Opening question:** {b.get('opening_question','').strip()}

"""
        if 'sample_dm' in b and b.get('sample_dm'):
            md += f"## Sample LinkedIn DM\n{b['sample_dm'].strip()}\n\n"
        if 'email_subject' in b and b.get('email_subject'):
            md += f"**Email subject:** {b['email_subject'].strip()}\n\n"
        if 'meeting_agenda' in b and b.get('meeting_agenda'):
            md += f"## Suggested Agenda\n{join_bullets('meeting_agenda')}\n"
        return md

    # If model returned raw text, display as-is
    if "raw_text" in brief:
        st.subheader("Brief (Unstructured)")
        st.write(brief["raw_text"])
        md_content = brief["raw_text"]
    else:
        st.subheader("Brief")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Snapshot:**\n\n{brief.get('profile_snapshot','')}")
            st.markdown("**Top Skills:**")
            st.write(brief.get("top_skills", []))
            st.markdown("**Domain Knowledge:**")
            st.write(brief.get("domain_knowledge", []))
            st.markdown("**Mutual Interests:**")
            st.write(brief.get("mutual_interests", []))
        with cols[1]:
            st.markdown("**Talking Points:**")
            st.write(brief.get("talking_points", []))
            st.markdown("**Icebreakers:**")
            st.write(brief.get("icebreakers", []))
            st.markdown("**Opening Question:**")
            st.write(brief.get("opening_question", ""))
            if 'sample_dm' in brief:
                st.markdown("**Sample LinkedIn DM:**")
                st.write(brief.get("sample_dm", ""))
            if 'email_subject' in brief:
                st.markdown("**Email Subject:**")
                st.write(brief.get("email_subject", ""))
            if 'meeting_agenda' in brief:
                st.markdown("**Suggested Agenda:**")
                st.write(brief.get("meeting_agenda", []))

        md_content = as_markdown(brief)

    st.download_button("⬇️ Download as Markdown", data=md_content, file_name="connection_prep_brief.md")
    st.caption("Tip: You can paste the Markdown into Notion/Docs or convert to PDF.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + Transformers (open-source).")
