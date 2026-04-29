"""
RAG CHAIN — The brain of the Q&A system.

What this file does:
  1. Loads the FAISS index that ingest.py built
  2. Creates a "retriever" that finds the most relevant chunks for any question
  3. Builds an LCEL chain:  question → retrieve context → format prompt → LLM → answer

The chain flow:
    User question
         │
         ▼
    Retriever (searches FAISS for top-k matching chunks)
         │
         ▼
    format_docs() (joins chunks into one readable string)
         │
         ▼
    ChatPromptTemplate (wraps context + question into a proper prompt)
         │
         ▼
    ChatOpenAI (sends prompt to GPT, streams back answer)
         │
         ▼
    StrOutputParser (converts LLM message object → plain string)

Run from project root to test in terminal:
    python src/rag_chain.py
    (requires ingest.py to have been run first)
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API keys from .env (OPENAI_API_KEY + optional LANGSMITH keys)
load_dotenv()

# Path to the FAISS index folder built by ingest.py
INDEX_PATH = str(Path(__file__).parent.parent / "faiss_index")

# System prompt tells the LLM how to behave.
# {context} is a placeholder — it gets filled with retrieved chunks at runtime.
# Instructing "say I don't know" prevents hallucination (making up answers).
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided context. If the context does not contain enough information \
to answer, say "I don't know based on the provided documents."

Context:
{context}"""


def load_retriever(index_path: str = INDEX_PATH, k: int = 4):
    """
    Load the saved FAISS index from disk and return a retriever object.

    The retriever's job: given a question, find the k most similar chunks.
    It converts the question to an embedding vector, then searches FAISS
    for the nearest vectors (= most relevant content).

    Args:
        index_path: path to the faiss_index/ folder (built by ingest.py)
        k:          how many chunks to retrieve per question
                    higher k = more context, but also more tokens sent to LLM

    Returns:
        retriever: a LangChain retriever object, usable in a chain

    Note:
        allow_dangerous_deserialization=True is required by FAISS when loading
        from disk — it's safe here because we created the index ourselves.
    """
    # Use the same embedding model as ingest.py — must match or search breaks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the index files from disk into memory
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    # as_retriever() wraps the vectorstore so LangChain chains can use it
    # search_kwargs={"k": k} means "return top k results"
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    """
    Convert a list of Document objects into one formatted string.

    The retriever returns a list of Document objects. The LLM needs
    a single string. This function joins them with clear separators
    and labels so the LLM knows where each source starts/ends.

    Example output:
        [Source 1]
        Task decomposition is the process of...

        ---

        [Source 2]
        Chain-of-thought prompting breaks down...

    Args:
        docs: list of Document objects from the retriever

    Returns:
        str: all chunk contents joined into one readable block
    """
    return "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )


def build_chain(retriever=None):
    """
    Assemble the full RAG chain using LangChain's LCEL (pipe | syntax).

    LCEL chains work like Unix pipes — each step's output feeds the next.
    This is LangChain Expression Language, the modern way to build chains.

    Chain breakdown:
        {
          "context":  retriever | format_docs,  ← runs retriever, then formats results
          "question": RunnablePassthrough()      ← passes the question through unchanged
        }
        | prompt      ← fills {context} and {question} into the prompt template
        | llm         ← sends filled prompt to GPT-4o-mini
        | StrOutputParser()  ← extracts the text from the LLM's response object

    RunnablePassthrough() means "don't transform this input, just pass it along".
    It lets the question flow into both the retriever AND the prompt template.

    Args:
        retriever: optional pre-built retriever (used by app.py to pass custom k)
                   if None, loads default retriever from disk

    Returns:
        chain: a runnable LangChain chain. Call with:
               chain.invoke("your question")          ← returns full string
               chain.stream("your question")          ← yields chunks (for streaming UI)
    """
    # If no retriever passed in, load with defaults
    if retriever is None:
        retriever = load_retriever()

    # Build the prompt template.
    # from_messages takes a list of (role, content) tuples.
    # "system" = instructions to the LLM
    # "human"  = the user's actual question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # temperature=0 → deterministic answers (no randomness).
    # Good for Q&A where you want factual, consistent responses.
    # gpt-4o-mini is fast and cheap — good for portfolio demos.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Assemble the chain using the | (pipe) operator
    # Each | passes output of left side as input to right side
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Entry point ────────────────────────────────────────────────────────────────
# Run this file directly to test the chain in your terminal.
# Requires faiss_index/ to exist (run ingest.py first).
if __name__ == "__main__":
    chain = build_chain()
    question = "What is task decomposition?"
    print(f"Q: {question}\n")

    # .stream() yields the answer word-by-word (like ChatGPT typing effect)
    # end="" and flush=True print each chunk immediately without newlines between
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()  # final newline after answer completes
