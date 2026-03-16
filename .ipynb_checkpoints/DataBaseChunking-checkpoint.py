"""
DataBaseChunking.py

Reads data_clean/*.txt → chunks by pharaoh → embeds → stores in ChromaDB
One chunk per pharaoh = clean Wikipedia paragraphs, no broken field extraction.
LLM formats the answer at query time.

Install:
    pip install langchain langchain-huggingface langchain-community
                chromadb sentence-transformers scikit-learn matplotlib

Run:
    python DataBaseChunking.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sklearn.manifold import TSNE

DATA_DIR   = Path("data_clean")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREVIEW_N   = 3

DYNASTY_COLORS = {
    "early dynastic": "#e74c3c", "dynasty 3":  "#e67e22",
    "dynasty 4":      "#f39c12", "dynasty 5":  "#d4ac0d",
    "dynasty 6":      "#a9cce3", "dynasty 11": "#2ecc71",
    "dynasty 12":     "#27ae60", "dynasty 15": "#c0392b",
    "dynasty 17":     "#922b21", "dynasty 18": "#3498db",
    "dynasty 19":     "#2980b9", "dynasty 20": "#1a5276",
    "dynasty 21":     "#8e44ad", "dynasty 22": "#7d3c98",
    "dynasty 25":     "#1abc9c", "dynasty 26": "#16a085",
    "dynasty 27":     "#7f8c8d", "dynasty 30": "#2c3e50",
    "ptolemaic":      "#e91e63",
}


# ── Config ─────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 has a 256-token limit (~1024 chars).
# 512-char body + ~80-char header = ~600 chars total, well within limits.
# Smaller chunks = crisper embeddings = better cosine similarity at query time.
CHUNK_SIZE    = 512   # body chars per chunk (header is extra ~80 chars)
CHUNK_OVERLAP = 80    # one short sentence of overlap keeps context


# ── Parse clean files with context-aware splitting ─────────────────────────
def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of ~size chars at paragraph boundaries.
    Each chunk overlaps the previous by ~overlap chars.
    Never cuts mid-paragraph.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks     = []
    current    = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > size and current:
            chunks.append("\n\n".join(current))
            # Keep last paragraph as overlap for next chunk
            overlap_paras = []
            overlap_len   = 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current     = overlap_paras
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def parse_clean_files(data_dir: Path) -> list[Document]:
    docs = []

    for txt_file in sorted(data_dir.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")

        first_line = content.split("\n")[0]
        dynasty    = first_line.replace("DYNASTY:", "").strip()

        blocks = re.split(r'\nPHARAOH:', content)

        for block in blocks[1:]:
            lines     = block.strip().split("\n")
            name      = lines[0].strip()
            src_match = re.search(r'SOURCE:\s*(.+)', block)
            source    = src_match.group(1).strip() if src_match else ""

            body_match = re.search(r'-{3,}\n+([\s\S]+?)(?:={3,}|$)', block)
            if not body_match:
                continue

            body = body_match.group(1).strip()
            if len(body) < 50:
                continue

            # Context header stamped on EVERY chunk.
            # Putting the pharaoh + dynasty in natural language makes the
            # embedding semantically richer than bare key:value labels.
            header = (
                f"About {name} ({dynasty}):\n"
                f"SOURCE: {source}\n"
            )

            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

            # ── Chunk 0: guaranteed intro chunk ───────────────────────────
            # Single first paragraph, capped at CHUNK_SIZE so it stays within
            # the embedding model's 256-token limit.
            # "Who is X?" queries always hit this chunk first.
            intro_body = paragraphs[0][:CHUNK_SIZE]
            rest_paras = paragraphs[1:]

            docs.append(Document(
                page_content=f"{header}\n{intro_body}",
                metadata={
                    "pharaoh":  name,
                    "dynasty":  dynasty,
                    "source":   source,
                    "file":     txt_file.name,
                    "chunk_n":  0,
                    "is_intro": "true",
                }
            ))

            # ── Remaining chunks: paragraph-boundary splits ────────────────
            if rest_paras:
                rest_body  = "\n\n".join(rest_paras)
                sub_chunks = split_into_chunks(rest_body, CHUNK_SIZE, CHUNK_OVERLAP)

                for i, chunk_body in enumerate(sub_chunks, start=1):
                    docs.append(Document(
                        page_content=f"{header}\n{chunk_body}",
                        metadata={
                            "pharaoh":  name,
                            "dynasty":  dynasty,
                            "source":   source,
                            "file":     txt_file.name,
                            "chunk_n":  i,
                            "is_intro": "false",
                        }
                    ))

    return docs


# ── Preview ────────────────────────────────────────────────────────────────
def preview(docs: list[Document]):
    print(f"\n{'='*65}")
    print(f"CHUNK PREVIEW — first {PREVIEW_N} of {len(docs)}")
    print(f"{'='*65}\n")

    for i, d in enumerate(docs[:PREVIEW_N]):
        print(f"── Chunk {i+1} ──────────────────────────────")
        print(f"  Pharaoh  : {d.metadata['pharaoh']}")
        print(f"  Dynasty  : {d.metadata['dynasty']}")
        print(f"  Chars    : {len(d.page_content)}")
        print(f"  Preview  :\n{d.page_content[:300]}")
        print()

    sizes = [len(d.page_content) for d in docs]
    print(f"── Stats ───────────────────────────────────")
    print(f"  Total chunks : {len(docs)}")
    print(f"  Min chars    : {min(sizes)}")
    print(f"  Max chars    : {max(sizes)}")
    print(f"  Avg chars    : {int(sum(sizes)/len(sizes))}\n")


# ── Embed + store ──────────────────────────────────────────────────────────
def store(docs: list[Document], embeddings) -> object:
    print("Embedding and storing in ChromaDB...")
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Saved to '{CHROMA_DIR}/'")
    return vs


# ── t-SNE visualization ────────────────────────────────────────────────────
def visualize_tsne(docs: list[Document], embeddings):
    print("\nGenerating t-SNE visualization...")

    texts     = [d.page_content for d in docs]
    labels    = [d.metadata["pharaoh"] for d in docs]
    dynasties = [d.metadata["dynasty"] for d in docs]

    print(f"  Embedding {len(texts)} pharaoh records...")
    vectors = np.array(embeddings.embed_documents(texts))

    print("  Running t-SNE...")
    tsne    = TSNE(n_components=2, perplexity=min(30, len(vectors)-1),
                   random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(vectors)

    def get_color(dynasty: str) -> str:
        d = dynasty.lower()
        for key, color in DYNASTY_COLORS.items():
            if key in d:
                return color
        return "#95a5a6"

    colors = [get_color(d) for d in dynasties]

    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor('#1a1208')
    ax.set_facecolor('#1a1208')
    ax.scatter(reduced[:, 0], reduced[:, 1],
               c=colors, s=90, alpha=0.85, edgecolors='white', linewidths=0.3)
    for i, (x, y) in enumerate(reduced):
        ax.annotate(labels[i], (x, y), fontsize=5.5, color='#f0e6cc',
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')

    ax.set_title("t-SNE: Pharaoh Embeddings",
                 color='#c9a84c', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE Dim 1", color='#7a6a50')
    ax.set_ylabel("t-SNE Dim 2", color='#7a6a50')
    ax.tick_params(colors='#7a6a50')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a2c10')

    seen = {}
    for dynasty, color in zip(dynasties, colors):
        k = dynasty[:35]
        if k not in seen:
            seen[k] = color
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=c, markersize=7, label=d)
               for d, c in sorted(seen.items())]
    ax.legend(handles=handles, loc='upper left', fontsize=6,
              framealpha=0.3, facecolor='#1a1208',
              edgecolor='#3a2c10', labelcolor='#f0e6cc')

    plt.tight_layout()
    out = Path("tsne_pharaohs.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1208')
    plt.close()
    print(f"  ✅ Saved: {out}")
    return out


# ── Retrieval test ─────────────────────────────────────────────────────────
def test(vs):
    queries = [
        "Who was Khufu?",
        "How old was Hatshepsut?",
        "What did Ramesses II build?",
        "How did Cleopatra die?",
        "Tell me about Djoser",
    ]
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    print(f"\n{'='*65}")
    print("RETRIEVAL TEST")
    print(f"{'='*65}\n")

    intro_queries = ["Who was Khufu?", "Tell me about Djoser"]

    for query in queries:
        print(f"Q: {query}")
        # Use intro filter for identity queries
        if any(t in query.lower() for t in ["who was", "tell me about", "who is"]):
            results = vs.similarity_search(query, k=2, filter={"is_intro": "true"})
        else:
            results = retriever.invoke(query)
        for r in results:
            print(f"  → [{r.metadata['pharaoh']}] "
                  f"{r.page_content[r.page_content.find(chr(10)+chr(10))+2:][:120].replace(chr(10),' ')}...")
        print()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.txt")):
        print(f"ERROR: No files in '{DATA_DIR}/'. Run clean_data.py first.")
        return

    print("Step 1: Loading clean files...")
    docs = parse_clean_files(DATA_DIR)
    for f in sorted(DATA_DIR.glob("*.txt")):
        count = sum(1 for d in docs if d.metadata["file"] == f.name)
        print(f"  {f.name}: {count} pharaohs")
    print(f"  → {len(docs)} total chunks\n")

    print("Step 2: Preview...")
    preview(docs)

    input("Chunks look good? Press Enter to embed + store, or Ctrl+C to abort: ")

    print("\nStep 3: Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("\nStep 4: Storing in ChromaDB...")
    vs = store(docs, embeddings)

    print("\nStep 5: t-SNE visualization...")
    tsne_file = visualize_tsne(docs, embeddings)

    print("\nStep 6: Testing retrieval...")
    test(vs)

    print(f"\n✅ Done.")
    print(f"   Chunks    : {len(docs)} pharaohs")
    print(f"   ChromaDB  : {CHROMA_DIR}/")
    print(f"   t-SNE map : {tsne_file}")


if __name__ == "__main__":
    main()