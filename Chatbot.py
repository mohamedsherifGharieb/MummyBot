"""
chatbot.py — RAG Chatbot with FAISS + Ollama

Install:
    pip install fastapi uvicorn faiss-cpu sentence-transformers ollama

Run:
    uvicorn chatbot:app --reload --port 8000
"""

import re
import pickle
import time
import numpy as np
import faiss
import ollama
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

INDEX_FILE  = Path("faiss_index.bin")
META_FILE   = Path("faiss_meta.pkl")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL   = "llama3.2"

SYSTEM_PROMPT = """
You are MummyBot — a friendly human-like guide at an ancient Egyptian museum.
You ONLY use the CONTEXT provided. Never add outside knowledge.

RULES:

1. If the question asks WHO someone is or for an introduction (who was, tell me about):
   Use this format:
   ## [Name]
   **Who:** one sentence
   **What they did:** one or two sentences
   **Key facts:**
   - fact from context
   - fact from context
   **In summary:** one plain sentence

2. If the question asks a SPECIFIC fact — how they died, what they built,
   how old they were, what battles they fought, who their family was:
   Skip the format. Answer in 1-2 plain conversational sentences directly.
   Like a tour guide talking to a tourist standing next to the exhibit.

3. If context does not contain the answer:
   "I don't have that in my scrolls yet! Try asking about another pharaoh."

4. If the question is not about ancient Egypt:
   "I only know about ancient Egyptian history! Ask me about a pharaoh."

Never say "Based on the context" or "According to the provided text".
Never repeat the full profile format for a specific question.
Just answer what was asked — directly and warmly.
"""

QUERY_TOPIC_MAP = {
    "IDENTITY":  ["who is", "who was", "tell me about", "what was"],
    "MONUMENTS": ["built", "build", "construct", "pyramid", "temple", "monument", "achieve"],
    "DEATH":     ["die", "died", "death", "killed", "how did", "cause", "buried", "tomb"],
    "REIGN":     ["rule", "reign", "campaign", "battle", "war", "fight", "battles"],
    "FAMILY":    ["children", "wife", "husband", "son", "daughter", "family", "married"],
    "LEGACY":    ["legacy", "famous", "known for", "remembered", "impact"],
}

PHARAOH_NAMES = sorted([
    "ramesses ii", "ramesses iii", "ramesses iv", "ramesses ix", "ramesses xi", "ramesses i",
    "thutmose iii", "thutmose iv", "thutmose ii", "thutmose i",
    "amenhotep iii", "amenhotep ii", "amenhotep i",
    "ptolemy iii", "ptolemy ii", "ptolemy i",
    "psamtik iii", "psamtik ii", "psamtik i",
    "pepi ii", "pepi i", "seti ii", "seti i",
    "mentuhotep iv", "mentuhotep iii", "mentuhotep ii",
    "senusret iii", "senusret ii", "senusret i",
    "amenemhat iii", "amenemhat ii", "amenemhat i",
    "nectanebo ii", "nectanebo i", "shoshenq iii", "shoshenq i",
    "khufu", "khafre", "menkaure", "snefru", "djedefre", "shepseskaf",
    "hatshepsut", "akhenaten", "tutankhamun", "horemheb", "nefertiti",
    "ahmose", "merneptah", "cleopatra vii", "cleopatra", "arsinoe",
    "djoser", "piye", "taharqa", "shabaka", "shebitku", "tantamani",
    "narmer", "tawosret", "kamose", "seqenenre",
    "cambyses", "darius", "xerxes", "artaxerxes",
    "osorkon", "psusennes", "pinedjem", "smendes",
    "userkaf", "sahure", "niuserre", "unas", "teti",
    "sobekneferu", "huni", "sekhemkhet",
], key=len, reverse=True)


print("Loading BGE model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Ready — {index.ntotal} vectors, {len(chunks)} chunks\n")


def detect_topic(query: str):
    q = query.lower()
    for topic, keywords in QUERY_TOPIC_MAP.items():
        if any(kw in q for kw in keywords):
            return topic
    return None


def extract_pharaoh(query: str):
    q = query.lower()
    for name in PHARAOH_NAMES:
        if name in q:
            roman = {"i","ii","iii","iv","v","vi","vii","viii","ix","x","xi"}
            return " ".join(p.upper() if p in roman else p.title()
                            for p in name.split())
    return None


def resolve_pronouns(query: str, history: list) -> str:
    pronouns = ["she ", "he ", "her ", "his ", "him ", "they "]
    q = query.lower()
    if not any(p in f" {q} " for p in pronouns):
        return query
    for turn in reversed(history[-6:]):
        content = turn.get("content", "").lower()
        for name in PHARAOH_NAMES:
            if name in content:
                roman  = {"i","ii","iii","iv","v","vi","vii","viii","ix","x","xi"}
                proper = " ".join(p.upper() if p in roman else p.title()
                                  for p in name.split())
                resolved = query
                for pronoun, replacement in [
                    ("she ", proper + " "), ("he ",  proper + " "),
                    ("her ", proper + "'s "), ("his ", proper + "'s "),
                    ("him ", proper + " "),
                ]:
                    resolved = re.sub(rf'\b{pronoun}', replacement + " ",
                                      resolved, flags=re.IGNORECASE)
                return resolved.strip()
    return query


def retrieve(query: str, k: int = 2) -> list:
    bge_q   = f"Represent this sentence for searching relevant passages: {query}"
    topic   = detect_topic(query)
    pharaoh = extract_pharaoh(query)

    q_vec = embed_model.encode([bge_q], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k * 10)
    candidates = [(chunks[i], float(scores[0][j]))
                  for j, i in enumerate(indices[0]) if i >= 0]

    if pharaoh:
        matched = [(c, s) for c, s in candidates
                   if c["pharaoh"].lower() == pharaoh.lower()]
        if matched:
            candidates = matched

    if topic:
        topic_matched = [(c, s) for c, s in candidates if c["topic"] == topic]
        if topic_matched:
            candidates = topic_matched

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in candidates[:k]]


app = FastAPI(title="MummyBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query:   str
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
    resolved = resolve_pronouns(req.query, req.history)
    docs     = retrieve(resolved, k=2)
    context  = "\n\n---\n\n".join(d["text"] for d in docs)
    sources  = list({d["source"] for d in docs})

    messages = [{"role": "system",
                 "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in req.history[-4:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    response = ollama.chat(
        model=LLM_MODEL, messages=messages,
        options={"num_predict": 400, "temperature": 0.2}
    )
    answer = response["message"]["content"].strip()
    return ChatResponse(answer=answer, sources=sources, retrieval_query=resolved)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    resolved = resolve_pronouns(req.query, req.history)
    docs     = retrieve(resolved, k=2)
    context  = "\n\n---\n\n".join(d["text"] for d in docs)

    messages = [{"role": "system",
                 "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in req.history[-4:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    print(f"Original : {req.query}")
    print(f"Resolved : {resolved}")
    print(f"Chunks   : {[(d['pharaoh'], d['topic']) for d in docs]}")

    def generate():
        stream = ollama.chat(
            model=LLM_MODEL, messages=messages, stream=True,
            options={"num_predict": 400, "temperature": 0.2}
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
                time.sleep(0.02)

    return StreamingResponse(generate(), media_type="text/plain")