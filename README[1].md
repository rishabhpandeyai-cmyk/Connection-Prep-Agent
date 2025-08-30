# Connection Prep Agent (Streamlit)

A free, legal, copy‑paste friendly app that turns a LinkedIn profile + recent posts into a 1‑page meeting brief.
No scraping, no automation. You only analyze text you can already view.

## Local Run

```bash
# 1) Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repo**.
2. Go to https://streamlit.io/cloud → **New app** → pick your repo.
3. Set **file** as `app.py`, **branch** `main`.
4. If you hit memory limits, switch the in-app engine to **Summarization‑only** or **Fast**.

## Legal & Privacy

- Do not scrape or automate LinkedIn; copy-paste only.
- Do not paste confidential information.
- On Streamlit Cloud, your data is processed on Streamlit's servers.
