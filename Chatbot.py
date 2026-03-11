"""
chatbot.py — RAG Pipeline + FastAPI

Install:
    pip install fastapi uvicorn langchain-huggingface langchain-community chromadb ollama

Run:
    uvicorn chatbot:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "llama3.2"
TOP_K       = 2   # chunks to retrieve per query

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are MummyBot, an ancient Egyptian history expert chatbot. You ONLY answer from the CONTEXT block provided in each message. You have ZERO outside knowledge.

STRICT RULES — follow every rule exactly:

RULE 1: If asked who you are, your name, or what you do:
  Respond: "I am MummyBot, your guide to ancient Egyptian history. Ask me about pharaohs, pyramids, mummies, and the wonders of the Nile!"
  Do NOT add anything else.

RULE 2: If the question is about ancient Egyptian history:
  a) Search ONLY the CONTEXT block for the answer.
  b) If the CONTEXT contains relevant information, answer using ONLY what is in the CONTEXT. Do NOT add any facts from your training data.
  c) Do NOT start with phrases like "You're asking about...", "Great question!", "Let me tell you..." or any introduction. Start your response directly with the answer.
  d) Format your answer in clean markdown:
     ## [Subject Name]
     **Overview:** 2-3 sentences using only context facts.
     **Key Facts:**
     - fact from context
     - fact from context
     **In Summary:** One plain sentence.
  e) If a specific detail (age, date, cause of death, etc.) is NOT stated in the CONTEXT, omit it entirely. Do NOT guess or invent it.

RULE 3: If the CONTEXT does not contain enough information to answer the question:
  Respond exactly: "I don't have that in my scrolls yet. Try asking about pharaohs, pyramids, mummification, or the Nile!"
  Do NOT attempt to answer from memory.

RULE 4: If the question has nothing to do with ancient Egyptian history (programming, sports, math, other countries, etc.):
  Respond exactly: "I only know about ancient Egyptian history! Ask me about pharaohs, pyramids, mummies, or the Nile."
  Do NOT answer the question.
"""

# ── Load vectorstore once at startup ─────────────────────────────────────────
print("Loading ChromaDB and embedding model...")
embeddings   = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)
vectorstore  = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
retriever    = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print("✅ Ready.\n")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Egypt Pharaoh Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"status": "Pharaoh Bot is running 🏺"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    docs    = retriever.invoke(req.query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "") for d in docs})

    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nTOURIST QUESTION: {req.query}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"].strip()

    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    docs    = retriever.invoke(req.query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    for msg in req.history[-4:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.query})

    def generate():
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
                time.sleep(0.025)

    return StreamingResponse(generate(), media_type="text/plain")