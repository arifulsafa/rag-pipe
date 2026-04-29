"""
STREAMLIT UI — RAG Q&A front-end.
Design system: ElevenLabs editorial (DESIGN.md)
  - Canvas #f5f5f5, ink #0c0a09, surface-card #ffffff
  - EB Garamond 300 for display, Inter 400/500 for body
  - Ink pill CTAs, hairline borders, atmospheric pastel orbs as decoration
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import load_from_url, load_from_file, build_index, INDEX_PATH
from rag_chain import build_chain, load_retriever

st.set_page_config(page_title="RAG Q&A", page_icon="", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@300;400&family=Inter:wght@400;500;600&display=swap');

/* ── Tokens ── */
:root {
  --canvas:        #f5f5f5;
  --canvas-soft:   #fafafa;
  --surface-card:  #ffffff;
  --surface-strong:#f0efed;
  --ink:           #0c0a09;
  --primary:       #292524;
  --body:          #4e4e4e;
  --muted:         #777169;
  --muted-soft:    #a8a29e;
  --hairline:      #e7e5e4;
  --hairline-strong:#d6d3d1;
  --success:       #16a34a;
  --error:         #dc2626;
  --orb-mint:      #a7e5d3;
  --orb-peach:     #f4c5a8;
  --orb-lavender:  #c8b8e0;
  --orb-sky:       #a8c8e8;
  --orb-rose:      #e8b8c4;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--canvas) !important;
  color: var(--ink);
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Layout ── */
.block-container {
  padding-top: 3rem;
  padding-bottom: 3rem;
  max-width: 900px;
}

/* ── Atmospheric orb behind hero (decoration only) ── */
.hero-orb {
  position: fixed;
  top: -180px;
  right: -120px;
  width: 520px;
  height: 520px;
  border-radius: 9999px;
  background: radial-gradient(circle, rgba(167,229,211,0.28) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}
.hero-orb-2 {
  position: fixed;
  bottom: -140px;
  left: -100px;
  width: 420px;
  height: 420px;
  border-radius: 9999px;
  background: radial-gradient(circle, rgba(200,184,224,0.22) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* ── Display type (EB Garamond 300) ── */
h1 {
  font-family: 'EB Garamond', 'Times New Roman', serif !important;
  font-weight: 300 !important;
  font-size: 2.6rem !important;
  line-height: 1.08 !important;
  letter-spacing: -0.04em !important;
  color: var(--ink) !important;
  margin-bottom: 0.25rem !important;
}
h2, h3 {
  font-family: 'EB Garamond', serif !important;
  font-weight: 300 !important;
  color: var(--ink) !important;
}

/* ── Caption ── */
.stCaption p {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.015em !important;
  color: var(--muted) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--surface-card) !important;
  border-right: 1px solid var(--hairline) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  color: var(--ink) !important;
}

/* ── Ink pill button (button-primary) ── */
.stButton > button {
  background-color: var(--primary) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 9999px !important;         /* pill */
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9375rem !important;          /* 15px */
  font-weight: 500 !important;
  padding: 10px 20px !important;
  height: 40px !important;
  letter-spacing: 0 !important;
  transition: background-color 0.15s ease;
}
.stButton > button:hover {
  background-color: var(--ink) !important;
}
.stButton > button:active {
  background-color: var(--ink) !important;
}

/* ── Text inputs (text-input spec: rounded-md 8px, h44, hairline-strong border) ── */
.stTextInput > div > div > input {
  background-color: var(--surface-card) !important;
  border: 1px solid var(--hairline-strong) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.016em !important;
  color: var(--ink) !important;
  height: 44px !important;
  padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
  border: 2px solid var(--ink) !important;
  box-shadow: none !important;
}

/* ── Radio buttons ── */
.stRadio label {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.016em !important;
  color: var(--body) !important;
}

/* ── Slider ── */
.stSlider label {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8125rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.015em !important;
}

/* ── Divider ── */
hr { border-color: var(--hairline) !important; }

/* ── Alert boxes ── */
.stAlert {
  border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
  background-color: var(--surface-card) !important;
  border: 1px solid var(--hairline-strong) !important;
  border-radius: 9999px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.016em !important;
  color: var(--ink) !important;
  padding: 12px 20px !important;
}
[data-testid="stChatInput"] textarea:focus {
  border: 2px solid var(--ink) !important;
  box-shadow: none !important;
}

/* ── User chat bubble — surface-strong tinted card ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background-color: var(--surface-strong) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: 16px !important;
  padding: 1rem 1.25rem !important;
  margin-bottom: 0.75rem !important;
}

/* ── Assistant chat bubble — white card with soft drop ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background-color: var(--surface-card) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: 16px !important;
  padding: 1rem 1.25rem !important;
  margin-bottom: 0.75rem !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
}

/* ── Chat message body text ── */
[data-testid="stChatMessage"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9375rem !important;
  font-weight: 400 !important;
  line-height: 1.5 !important;
  letter-spacing: 0.016em !important;
  color: var(--body) !important;
}

/* ── Info box ── */
[data-testid="stInfoBox"] {
  background-color: var(--canvas-soft) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: 12px !important;
  color: var(--muted) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
}
</style>

<!-- Atmospheric gradient orbs — decoration only, no content -->
<div class="hero-orb"></div>
<div class="hero-orb-2"></div>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("RAG Q&A")
st.caption("Ask questions over any document or web page")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")

    source_type = st.radio("Source", ["URL", "Text file / PDF"])

    if source_type == "URL":
        url = st.text_input(
            "Web URL",
            value="https://lilianweng.github.io/posts/2023-06-23-agent/",
        )
        if st.button("Load & Index"):
            if url:
                with st.spinner("Indexing…"):
                    try:
                        docs = load_from_url(url)
                        build_index(docs)
                        st.session_state["index_ready"] = True
                        st.success(f"Indexed {len(docs)} page(s)")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        uploaded = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
        if uploaded and st.button("Index file"):
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            with st.spinner("Indexing…"):
                try:
                    docs = load_from_file(str(tmp_path))
                    build_index(docs)
                    st.session_state["index_ready"] = True
                    st.success(f"Indexed {len(docs)} page(s)")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    if INDEX_PATH.exists() and "index_ready" not in st.session_state:
        st.session_state["index_ready"] = True
        st.info("Index loaded from disk")

    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=4)

# ── Chat ───────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.get("index_ready"):
    st.info("Load a document from the sidebar to begin.")
else:
    if question := st.chat_input("Ask anything about your documents…"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                retriever = load_retriever(k=k)
                chain = build_chain(retriever)
                for chunk in chain.stream(question):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                placeholder.error(full_response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
