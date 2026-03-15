from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import re
import time
import difflib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama3.2"
RETRIEVE_K    = 3
HISTORY_LIMIT = 10

SYSTEM_PROMPT = """
You are MummyBot, a friendly ancient Egyptian history guide.

RULES (follow strictly):
1. Answer ONLY using facts found in the CONTEXT below.
2. NEVER invent, guess, or add facts not in the CONTEXT — not dates, not numbers, not events.
3. If the CONTEXT does not contain the answer, you MUST say:
   "I don't have that in my scrolls! Try asking about another pharaoh."
4. If the question is not about ancient Egypt, say:
   "I only know ancient Egyptian history!"

Format (when the CONTEXT has an answer):
## [Name]
**Who:** one sentence  **Reign / Dynasty:** one sentence
**Key facts:** 2-3 bullet points copied from the context
**Summary:** one plain sentence
"""


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


def fuzzy_match_pharaoh(text: str) -> str | None:
    text = text.lower()
    for name in PHARAOH_NAMES:
        if name in text:
            return name.title()
    words = re.findall(r'[a-z]{3,}', text)
    for word in words:
        hits = difflib.get_close_matches(word, PHARAOH_NAMES, n=1, cutoff=0.7)
        if hits:
            return hits[0].title()
    return None


def rewrite_query(query: str, history: list[dict]) -> tuple[str, str | None]:
    msg = query.strip().lower()

    needs_resolution = any(f" {p} " in f" {msg} " for p in PRONOUNS)
    resolved_name = None

    if needs_resolution and history:
        for turn in reversed(history[-HISTORY_LIMIT:]):
            content = turn.get("content", "").lower()
            match = fuzzy_match_pharaoh(content)
            if match:
                resolved_name = match
                break

    detected_field = None
    for field, keywords in FIELD_MAP.items():
        if any(kw in msg for kw in keywords):
            detected_field = field
            break

    name_found = fuzzy_match_pharaoh(msg)
    pharaoh = name_found or resolved_name

    IDENTITY_TRIGGERS = ["tell me about", "who is", "who was", "what was", "about"]
    is_identity = any(t in query.lower() for t in IDENTITY_TRIGGERS)

    if pharaoh and detected_field and not is_identity:
        field_label = detected_field.lower().replace("_", " ")
        return f"{pharaoh} {field_label}", pharaoh
    elif pharaoh:
        return pharaoh, pharaoh
    else:
        return query.strip(), None


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
print("✅ Ready.\n")


# ── Retrieve ───────────────────────────────────────────────────────────────
def retrieve(query: str, pharaoh: str | None = None) -> list:
    """
    When we know the pharaoh name (from query rewriter), fetch more candidates
    and filter to ONLY that pharaoh's chunks. This guarantees we return the
    right pharaoh even if the embedding similarity is imperfect.
    When pharaoh is unknown, plain similarity search.
    """
    if pharaoh:
        docs = vectorstore.similarity_search(query, k=15)
        filtered = [
            d for d in docs
            if pharaoh.lower() in d.page_content.lower()
        ]
        if filtered:
            return filtered[:RETRIEVE_K]

    return vectorstore.similarity_search(query, k=RETRIEVE_K)



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

    retrieval_query, pharaoh = rewrite_query(req.query, history)
    print(f"Original : {req.query}")
    print(f"Rewritten: {retrieval_query}  (pharaoh={pharaoh})")

    docs    = retrieve(retrieval_query, pharaoh)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "") for d in docs})

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    response = ollama.chat(model=LLM_MODEL, messages=messages, options={"num_predict": 256})
    answer   = response["message"]["content"].strip()

    return ChatResponse(answer=answer, sources=sources, retrieval_query=retrieval_query)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history = req.history[-HISTORY_LIMIT:]

    retrieval_query, pharaoh = rewrite_query(req.query, history)
    print(f"Original : {req.query}")
    print(f"Rewritten: {retrieval_query}  (pharaoh={pharaoh})")

    docs    = retrieve(retrieval_query, pharaoh)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    def generate():
        stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True,
                             options={"num_predict": 256})
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    return StreamingResponse(generate(), media_type="text/plain")