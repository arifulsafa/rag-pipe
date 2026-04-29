# RAG Q&A System
# My Network URL - http://192.168.0.30:8501

Ask questions over any web page or document using Retrieval-Augmented Generation.

**Stack:** LangChain · OpenAI · FAISS · Streamlit · LangSmith

---

## Setup

```bash
cd rag-qa-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY (and optionally LANGCHAIN_API_KEY) to .env
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
| `src/ingest.py` | Load → split → embed → save FAISS index |
| `src/rag_chain.py` | Load index → build LangChain LCEL chain |
| `app.py` | Streamlit UI with sidebar document loader |

## LangSmith Tracing (optional)

Set in `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT="rag-qa-app"
```
Every run appears at https://smith.langchain.com with full retrieval traces.

## Extensions to explore

- **Multi-query retrieval** — `MultiQueryRetriever` generates query variants, improves recall
- **Hybrid search** — combine FAISS with BM25 (`EnsembleRetriever`)
- **Conversational memory** — `RunnableWithMessageHistory` for follow-up questions
- **Re-ranking** — Cohere or cross-encoder reranker after retrieval
- **LangGraph** — turn this into a stateful agent graph
