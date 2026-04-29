# RAG Q&A System

Ask questions over any web page or document using Retrieval-Augmented Generation.

**Stack:** LangChain · OpenAI · FAISS · Streamlit · LangSmith

---

## Setup

```bash
cd rag-qa-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY (and optionally LANGSMITH_API_KEY) to .env
```

## Run

### Step 1 — Index a document (CLI)
```bash
python src/ingest.py
# Indexes https://lilianweng.github.io/posts/2023-06-23-agent/ by default
```

### Step 2 — Ask questions (CLI)
```bash
python src/rag_chain.py
```

### Step 3 — Streamlit UI
```bash
streamlit run app.py
```
Open http://localhost:8501, paste any URL in the sidebar, click **Load & Index**, then ask away.

> **Note:** If localhost doesn't connect, use your network URL (printed in terminal on startup).

---

## Architecture

```
User question
     │
     ▼
  Retriever (FAISS similarity search, k=4 chunks)
     │
     ▼
  ChatPromptTemplate  ←  retrieved context
     │
     ▼
  ChatOpenAI (gpt-4o-mini)
     │
     ▼
  Streamed answer
```

## Key files

| File | Purpose |
|------|---------|
| `src/ingest.py` | Load → split → embed → save/merge FAISS index |
| `src/rag_chain.py` | Load index → build LangChain LCEL chain |
| `app.py` | Streamlit UI with sidebar document loader |
| `.streamlit/config.toml` | Forces light theme |
| `.streamlit/secrets.toml` | API keys for Streamlit Cloud (gitignored) |

## Multi-document support

Uploading a new document **merges** into the existing index — previous docs are preserved.
To reset and start fresh:
```bash
rm -rf faiss_index/
```

## LangSmith Tracing (optional)

Set in `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT="rag-qa-app"
```
Every run appears at https://smith.langchain.com with full retrieval traces.

## Deploy to Streamlit Cloud

1. Push repo to GitHub (`.env` and `.streamlit/secrets.toml` are gitignored)
2. Go to share.streamlit.io → connect repo → set main file to `app.py`
3. Paste API keys into the **Secrets** dashboard (same format as `secrets.toml`)
4. Deploy — live URL in ~2 minutes

## Extensions to explore

- **Multi-query retrieval** — `MultiQueryRetriever` generates query variants, improves recall
- **Hybrid search** — combine FAISS with BM25 (`EnsembleRetriever`)
- **Conversational memory** — `RunnableWithMessageHistory` for follow-up questions
- **Re-ranking** — Cohere or cross-encoder reranker after retrieval
- **LangGraph** — turn this into a stateful agent graph
