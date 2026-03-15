# 🏺 MummyBot — Ancient Egypt Museum Guide

A RAG-powered chatbot built to help tourists explore ancient Egyptian history during their museum visit. Ask about pharaohs, mummies, dynasties, pyramids, and the stories behind the artifacts you see on display.

---

## What Is This

MummyBot is a personal learning project built around Retrieval-Augmented Generation. The goal was to understand how RAG works end to end — from scraping raw data to chunking, embedding, retrieval, and finally serving answers through a local LLM.

The theme is ancient Egypt because it makes a genuinely useful museum companion. Instead of reading every wall plaque, a tourist can ask natural questions and get contextual answers tied to real historical sources.

---

## What It Can Do

- Answer questions about 86 pharaohs across 19 dynasties
- Tell you who a pharaoh was, what they built, how they died, and what era they ruled in
- Handle follow-up questions using conversation history
- Stream answers token by token so you are not staring at a blank screen
- Show you which Wikipedia source the answer came from

---

## Pharaohs Covered

| Era | Dynasties |
|---|---|
| Early Dynastic | 1–2 |
| Old Kingdom | 3, 4, 5, 6 |
| Middle Kingdom | 11, 12 |
| Second Intermediate | 15 (Hyksos), 17 (Theban) |
| New Kingdom | 18, 19, 20 |
| Third Intermediate | 21, 22 |
| Nubian | 25 |
| Late Period | 26, 27 (Persian), 30 |
| Ptolemaic | Cleopatra VII through Ptolemy III |

---

## Stack

| Component | Tool |
|---|---|
| Embedding | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Retrieval | MMR (Maximum Marginal Relevance) |
| LLM |  Ollama |
| Backend | FastAPI + uvicorn |
| Frontend | Angular 21 (standalone components, signals) |

Everything runs locally. No API keys. No cloud costs.

---

## How RAG Works Here

```
Tourist asks: "How did Cleopatra die?"
        ↓
Query rewriter detects pharaoh name + intent
        ↓
ChromaDB searches 469 embedded chunks
        ↓
MMR retrieval picks 2 diverse relevant chunks
        ↓
Phi-3 Mini reads chunks + formats answer
        ↓
Answer streams back to the Angular frontend
```

The LLM never generates from memory. Every answer is grounded in the scraped Wikipedia source text stored in ChromaDB.

---

## Project Structure

```
AiHistroicChatbot/
├── egypt_scraper.py        # scrapes Wikipedia for 86 pharaohs
├── clean_data.py           # strips citations, markup, pronunciation guides
├── DataBaseChunking.py     # chunks by pharaoh + embeds into ChromaDB
├── chatbot.py              # FastAPI backend — RAG pipeline
├── data_clean/             # cleaned text files, one per dynasty
├── chroma_db/              # vector database
├── tsne_pharaohs.png       # 2D visualization of embedding space
└── Frontend/               # Angular chatbot UI
    └── egypt-chatbot/
        └── src/app/
            ├── app.component.ts
            ├── app.component.html
            ├── app.component.scss
            ├── chat-message/
            ├── chat-input/
            └── pyramid-loader/
```

---

## Run It Yourself

**Requirements**

- Python 3.10+
- Node.js 18+
- Ollama installed and running

**Install dependencies**

```bash
pip install fastapi uvicorn langchain-huggingface langchain-community
            chromadb sentence-transformers scikit-learn matplotlib
            beautifulsoup4 requests
```

**Pull the model**

```bash
ollama pull phi3:mini
```

**Build the database**

```bash
python egypt_scraper.py        # scrapes Wikipedia → egypt_raw_data.md
python clean_data.py           # cleans text → data_clean/
python DataBaseChunking.py     # embeds chunks → chroma_db/
```

**Start the backend**

```bash
uvicorn chatbot:app --reload --port 8000
```

**Start the frontend**

```bash
cd Frontend/egypt-chatbot
npm install
ng serve
```

Open `http://localhost:4200`

---

## What I Learned Building This

**Chunking strategy matters more than the LLM.** A 3B model with clean, well-scoped chunks beats a 70B model handed a wall of text. The breakthrough was separating each pharaoh into a guaranteed intro chunk and content chunks so identity queries always hit the right context first.

**Rerankers are expensive on CPU.** CrossEncoder was adding 8+ seconds per query. Switching to MMR handled result diversity without the inference overhead.

**Small embedding models hit a token ceiling.** all-MiniLM-L6-v2 maxes out at 256 tokens meaning chunks over roughly 1000 characters get silently truncated. Paragraph-boundary splitting kept chunks within the window.

**t-SNE revealed how well the embeddings clustered.** Ptolemaic pharaohs sat isolated from native Egyptians. Ramesside kings grouped tightly. Akhenaten ended up alone because his Wikipedia article is dominated by religious and heresy language, semantically distant from military pharaohs.

---

## Example Questions to Ask

```
Who was Hatshepsut?
How did Cleopatra die?
What did Ramesses II build?
Tell me about Tutankhamun
Which pharaoh built the Step Pyramid?
Who were the Nubian pharaohs?
What happened to Egypt after Cleopatra?
```

---

## Data Sources

All pharaoh data is sourced from Wikipedia under the Creative Commons Attribution-ShareAlike License. MummyBot does not generate facts — it only retrieves and summarizes from these sources.
