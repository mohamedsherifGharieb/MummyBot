"""
chatbot.py — RAG Pipeline + FastAPI

Install:
    pip install fastapi uvicorn langchain-huggingface langchain-community
                chromadb ollama sentence-transformers

Run:
    uvicorn chatbot:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import re
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
RETRIEVE_K    = 10   # cast wide net
RERANK_TOP_N  = 2    # reranker picks best 3
HISTORY_LIMIT = 4

SYSTEM_PROMPT = """
You are MummyBot — a friendly ancient Egyptian history guide for tourists.
You answer questions using ONLY the CONTEXT provided. Never use outside knowledge.

If the question is about ancient Egypt, answer using the CONTEXT in this format:

## [Name]
**Who:** one sentence
**What they did:** one or two sentences
**Key facts:**
- fact from context
- fact from context
**In summary:** one plain sentence any tourist would understand

If the CONTEXT does not contain the answer, say:
"I don't have that in my scrolls yet! Try asking about another pharaoh."

If the question is not about ancient Egypt, say:
"I only know about ancient Egyptian history! Ask me about a pharaoh."

Keep answers friendly, clear, and based only on the CONTEXT.
"""

# ── Pharaoh name list for query rewriting ──────────────────────────────────
PHARAOH_NAMES = [
    "khufu", "khafre", "menkaure", "snefru", "djedefre", "shepseskaf",
    "hatshepsut", "thutmose", "amenhotep", "akhenaten", "tutankhamun",
    "horemheb", "nefertiti", "ay", "ahmose",
    "ramesses", "seti", "merneptah", "tawosret",
    "piye", "shabaka", "shebitku", "taharqa", "tantamani",
    "cleopatra", "ptolemy", "arsinoe",
    "narmer", "djoser", "pepi", "mentuhotep", "senusret",
    "amenemhat", "sobekneferu", "unas", "sahure", "userkaf",
    "nectanebo", "cambyses", "darius", "xerxes", "psamtik",
    "necho", "apries", "ahmose ii", "seqenenre", "kamose",
]

FIELD_MAP = {
    "AGE":           ["old", "age", "born", "birth", "years", "lived", "how long"],
    "HOW_THEY_DIED": ["die", "died", "death", "killed", "murder", "suicide", "poison", "cause"],
    "ACHIEVEMENTS":  ["build", "built", "construct", "achieve", "monument", "pyramid", "temple", "accomplish"],
    "WHAT_THEY_DID": ["do", "did", "rule", "reign", "known for", "famous for"],
    "WHO":           ["who", "tell me about", "what was", "about", "identity"],
    "EGYPT_ERA":     ["era", "period", "time", "context", "egypt like", "dynasty"],
}

PRONOUNS = ["he", "she", "they", "his", "her", "their", "him", "this pharaoh", "this ruler"]


# ── Query rewriter ─────────────────────────────────────────────────────────
def rewrite_query(query: str, history: list[dict]) -> str:
    msg = query.strip().lower()

    # Resolve pronouns using last pharaoh mentioned in history
    needs_resolution = any(f" {p} " in f" {msg} " for p in PRONOUNS)
    resolved_name = None

    if needs_resolution and history:
        for turn in reversed(history[-HISTORY_LIMIT:]):
            content = turn.get("content", "").lower()
            for name in PHARAOH_NAMES:
                if name in content:
                    resolved_name = name.title()
                    break
            if resolved_name:
                break

    # Detect field intent
    detected_field = None
    for field, keywords in FIELD_MAP.items():
        if any(kw in msg for kw in keywords):
            detected_field = field
            break

    # Extract pharaoh name from current message
    name_found = None
    for name in PHARAOH_NAMES:
        if name in msg:
            name_found = name.title()
            break

    pharaoh = name_found or resolved_name

    # "tell me about", "who is/was" = identity query, just return the name
    # Don't append field name — "Cleopatra who" is a bad embedding query
    IDENTITY_TRIGGERS = ["tell me about", "who is", "who was", "what was", "about"]
    is_identity = any(t in query.lower() for t in IDENTITY_TRIGGERS)

    if pharaoh and detected_field and not is_identity:
        field_label = detected_field.lower().replace("_", " ")
        return f"{pharaoh} {field_label}"
    elif pharaoh:
        return pharaoh
    else:
        return query.strip()


# ── Load models at startup ─────────────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)

print("Loading ChromaDB...")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# Child-only retriever — precise field-level retrieval
child_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": RETRIEVE_K,
        "filter": {"chunk_type": "field_chunk"}
    }
)

# Fallback retriever — no filter, used when child returns nothing
fallback_retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVE_K}
)

print("✅ Ready.\n")


# ── No reranker — use MMR diversity in retrieval instead ──────────────────
def rerank(query: str, docs: list) -> list:
    return docs[:2]   # already MMR-ranked, just take top 2


# ── Query → topic mapping ──────────────────────────────────────────────────
QUERY_TOPIC_MAP = {
    "IDENTITY":  ["who is", "who was", "tell me about", "what was", "about"],
    "MONUMENTS": ["built", "build", "construct", "pyramid", "temple", "monument", "achieve"],
    "DEATH":     ["die", "died", "death", "killed", "how did", "cause", "buried", "tomb"],
    "REIGN":     ["did", "do", "rule", "reign", "campaign", "battle", "war", "accomplish"],
    "FAMILY":    ["children", "wife", "husband", "son", "daughter", "family", "married"],
    "LEGACY":    ["legacy", "famous", "known for", "remembered", "impact", "why"],
}


def detect_topic(query: str) -> str | None:
    q = query.lower()
    for topic, keywords in QUERY_TOPIC_MAP.items():
        if any(kw in q for kw in keywords):
            return topic
    return None


# ── Retrieve using topic filter + MMR ──────────────────────────────────────
def retrieve(query: str) -> list:
    topic = detect_topic(query)

    if topic:
        # Filter to the matching topic first — precise retrieval
        docs = vectorstore.similarity_search(
            query, k=4, filter={"topic": topic}
        )
        if docs:
            return docs[:2]

    # Fallback: MMR across all chunks
    docs = vectorstore.max_marginal_relevance_search(
        query, k=2, fetch_k=10
    )
    return docs


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Egypt Pharaoh Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer:          str
    sources:         list[str]
    retrieval_query: str


@app.get("/")
def root():
    return {"status": "MummyBot is running 🏺"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = req.history[-HISTORY_LIMIT:]

    retrieval_query = rewrite_query(req.query, history)
    print(f"Original : {req.query}")
    print(f"Rewritten: {retrieval_query}")

    docs    = retrieve(retrieval_query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "") for d in docs})

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"num_predict": 400, "temperature": 0.2}
    )
    answer   = response["message"]["content"].strip()

    return ChatResponse(answer=answer, sources=sources, retrieval_query=retrieval_query)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    import time
    history = req.history[-HISTORY_LIMIT:]

    t0 = time.time()
    retrieval_query = rewrite_query(req.query, history)
    print(f"Original : {req.query}")
    print(f"Rewritten: {retrieval_query}")

    t1 = time.time()
    docs    = retrieve(retrieval_query)
    t2 = time.time()
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    print(f"⏱ rewrite={t1-t0:.2f}s  retrieve+rerank={t2-t1:.2f}s  chunks={len(docs)}")

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    def generate():
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            options={"num_predict": 400, "temperature": 0.2}
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
                time.sleep(0.02)

    return StreamingResponse(generate(), media_type="text/plain")