"""
INGEST PIPELINE — Run this ONCE before using the chatbot.

What this file does:
  1. Loads text from a URL or file
  2. Splits it into small chunks (so the AI can search precisely)
  3. Converts each chunk into numbers called "embeddings" (OpenAI does this)
  4. Saves all those embeddings into a FAISS index on disk

Think of it like scanning a book, cutting it into index cards,
writing a topic-code on each card, and filing them in a cabinet.
Later, rag_chain.py opens that cabinet to find relevant cards for each question.

Run from project root:
    python src/ingest.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API keys from .env file (OPENAI_API_KEY must be set there)
load_dotenv()

# Where the FAISS index folder will be saved on disk.
# Path(__file__) = this file's location (src/ingest.py)
# .parent.parent   = go up two levels → project root
# / "faiss_index"  = folder name
INDEX_PATH = Path(__file__).parent.parent / "faiss_index"


def load_from_url(url: str):
    """
    Fetch and parse a web page.

    WebBaseLoader uses BeautifulSoup under the hood to strip HTML tags
    and return clean text. Returns a list of Document objects —
    each Document has .page_content (the text) and .metadata (url, title, etc).

    Example:
        docs = load_from_url("https://example.com/blog-post")
        print(docs[0].page_content)  # raw text of the page
    """
    return WebBaseLoader(url).load()


def load_from_file(path: str):
    """
    Load text from a local .txt or .pdf file.

    - PDF  → PyPDFLoader: extracts text page by page
    - .txt → TextLoader:  reads the file as-is

    Returns a list of Document objects, same format as load_from_url.

    Example:
        docs = load_from_file("my_notes.txt")
        docs = load_from_file("research_paper.pdf")
    """
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path).load()


def build_index(docs, index_path: str = str(INDEX_PATH)):
    """
    Takes loaded documents, splits them, embeds them, and saves a FAISS index.

    Step 1 — Split:
        RecursiveCharacterTextSplitter cuts text into chunks.
        chunk_size=1000   → each chunk is ~1000 characters
        chunk_overlap=150 → consecutive chunks share 150 chars (so context
                            doesn't get cut off at chunk boundaries)

        Why split? The LLM can't search one giant blob. Smaller, focused
        chunks give more precise search results.

    Step 2 — Embed:
        OpenAIEmbeddings sends each chunk to OpenAI's embedding API.
        It returns a list of 1536 numbers (a "vector") per chunk.
        Chunks with similar meaning get similar vectors — that's how
        semantic search works (not keyword matching).

        model="text-embedding-3-small" → cheaper, still very accurate

    Step 3 — Save:
        FAISS (Facebook AI Similarity Search) stores all those vectors
        in an efficient index on disk so we can search them instantly later.
        Saved to faiss_index/ folder in project root.

    Args:
        docs:       list of Document objects from load_from_url or load_from_file
        index_path: where to save the FAISS index folder (default: faiss_index/)

    Returns:
        vectorstore: the in-memory FAISS object (also saved to disk)
    """
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # max characters per chunk
        chunk_overlap=150,  # overlap between consecutive chunks
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Convert chunks to vectors using OpenAI's embedding model
    # This makes API calls to OpenAI — uses a small amount of credits
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # If an index already exists on disk, load it and MERGE new chunks into it.
    # This preserves all previously indexed documents.
    # If no index exists yet, create a fresh one from scratch.
    index_path_obj = Path(index_path)
    if index_path_obj.exists():
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
        print(f"Merged {len(chunks)} new chunks into existing index")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"Created new index with {len(chunks)} chunks")

    # Save (overwrite) the index on disk with the updated version
    vectorstore.save_local(index_path)
    print(f"Index saved → {index_path}")

    return vectorstore


# ── Entry point ────────────────────────────────────────────────────────────────
# This block only runs when you execute: python src/ingest.py
# It won't run when other files import this module (e.g. app.py)
if __name__ == "__main__":
    # Demo: index Lilian Weng's famous blog post about LLM agents
    # Change this URL (or swap for load_from_file) to index your own content
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    print(f"Loading: {url}")
    docs = load_from_url(url)
    build_index(docs)
