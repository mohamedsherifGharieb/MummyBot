"""
DataBaseChunking.py

Reads data_clean/*.txt → builds territory chunks → embeds with BGE → stores in FAISS

Territory strategy:
  - Each pharaoh gets one ANCHOR chunk (identity signal)
  - All chunks prefixed with [DynastyTag][PharaohName] → clusters in vector space
  - Stored as faiss_index.bin + faiss_meta.pkl (no LangChain, no ChromaDB)

Install:
    pip install faiss-cpu sentence-transformers transformers scikit-learn matplotlib

Run:
    python DataBaseChunking.py
"""

import re
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import faiss

DATA_DIR    = Path("data_clean")
INDEX_FILE  = Path("faiss_index.bin")
META_FILE   = Path("faiss_meta.pkl")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS  = 450
TOPIC_LABELS = ["IDENTITY", "REIGN", "MONUMENTS", "FAMILY", "DEATH", "LEGACY"]

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

tokenizer_ = AutoTokenizer.from_pretrained(EMBED_MODEL)


def count_tokens(text: str) -> int:
    return len(tokenizer_.encode(text, truncation=False))


def split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 30]


def trim_to_tokens(text: str, max_tok: int) -> str:
    sentences = split_sentences(text)
    result, total = [], 0
    for s in sentences:
        t = count_tokens(s)
        if total + t > max_tok:
            break
        result.append(s)
        total += t
    return " ".join(result)


def split_to_token_chunks(text: str, max_tok: int, prefix: str) -> list:
    sentences = split_sentences(text)
    chunks, current, cur_tok = [], [], 0
    for s in sentences:
        st = count_tokens(s)
        if cur_tok + st > max_tok and current:
            chunks.append(prefix + " ".join(current))
            current, cur_tok = [s], st
        else:
            current.append(s)
            cur_tok += st
    if current:
        chunks.append(prefix + " ".join(current))
    return chunks


def parse_clean_files(data_dir: Path) -> list:
    chunks = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        content    = txt_file.read_text(encoding="utf-8")
        dynasty    = content.split("\n")[0].replace("DYNASTY:", "").strip()
        dyn_tag    = re.sub(r'[^a-zA-Z0-9]', '', dynasty[:30])
        blocks     = re.split(r'\nPHARAOH:', content)

        for block in blocks[1:]:
            lines     = block.strip().split("\n")
            name      = lines[0].strip()
            src_match = re.search(r'SOURCE:\s*(.+)', block)
            source    = src_match.group(1).strip() if src_match else ""
            body_match = re.search(r'-{3,}\n+([\s\S]+?)(?:={3,}|$)', block)
            if not body_match:
                continue
            body      = body_match.group(1).strip()
            if len(body) < 50:
                continue

            territory = f"[{dyn_tag}][{name}] "

            # ── ANCHOR chunk — one per pharaoh ────────────────────────────
            all_sents = split_sentences(body)
            intro     = trim_to_tokens(" ".join(all_sents[:3]),
                                       MAX_TOKENS - count_tokens(territory) - 20)
            chunks.append({
                "text":      territory + intro,
                "pharaoh":   name,
                "dynasty":   dynasty,
                "topic":     "ANCHOR",
                "source":    source,
                "is_anchor": True,
            })

            # ── Topic baby chunks ─────────────────────────────────────────
            parts  = re.split(r'\[(' + '|'.join(TOPIC_LABELS) + r')\]\n', body)
            i = 0
            while i < len(parts) - 1:
                if parts[i].strip() in TOPIC_LABELS:
                    topic        = parts[i].strip()
                    content_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if len(content_text) > 40:
                        topic_prefix = territory + f"{topic}: "
                        max_body_tok = MAX_TOKENS - count_tokens(topic_prefix) - 5
                        if count_tokens(content_text) <= max_body_tok:
                            sub_chunks = [topic_prefix + content_text]
                        else:
                            sub_chunks = split_to_token_chunks(
                                content_text, max_body_tok, topic_prefix)
                        for sub in sub_chunks:
                            chunks.append({
                                "text":      sub,
                                "pharaoh":   name,
                                "dynasty":   dynasty,
                                "topic":     topic,
                                "source":    source,
                                "is_anchor": False,
                            })
                    i += 2
                else:
                    i += 1

    return chunks


def embed_and_store(chunks: list) -> tuple:
    print(f"Loading {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts, batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ FAISS index saved: {INDEX_FILE} ({INDEX_FILE.stat().st_size//1024} KB)")
    print(f"✅ Metadata saved  : {META_FILE}  ({META_FILE.stat().st_size//1024} KB)")
    return index, embeddings


def visualize_tsne(chunks: list, embeddings: np.ndarray):
    print("\nGenerating t-SNE territory map...")
    anchor_idx  = [i for i, c in enumerate(chunks) if c["is_anchor"]]
    anchor_vecs = embeddings[anchor_idx]
    labels      = [chunks[i]["pharaoh"] for i in anchor_idx]
    dynasties   = [chunks[i]["dynasty"] for i in anchor_idx]

    def get_color(d):
        dl = d.lower()
        for k, c in DYNASTY_COLORS.items():
            if k in dl: return c
        return "#95a5a6"

    colors  = [get_color(d) for d in dynasties]
    tsne    = TSNE(n_components=2, perplexity=min(30, len(anchor_vecs)-1),
                   random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(anchor_vecs)

    fig, ax = plt.subplots(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1208')
    ax.set_facecolor('#1a1208')
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=100,
               alpha=0.9, edgecolors='white', linewidths=0.4)
    for i, (x, y) in enumerate(reduced):
        ax.annotate(labels[i], (x, y), fontsize=5.5, color='#f0e6cc',
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')
    ax.set_title("Territory Map — One Anchor Per Pharaoh",
                 color='#c9a84c', fontsize=13, fontweight='bold')
    ax.set_xlabel("t-SNE Dim 1", color='#7a6a50')
    ax.set_ylabel("t-SNE Dim 2", color='#7a6a50')
    ax.tick_params(colors='#7a6a50')
    for spine in ax.spines.values(): spine.set_edgecolor('#3a2c10')
    seen = {}
    for d, c in zip(dynasties, colors):
        if d[:35] not in seen: seen[d[:35]] = c
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=c, markersize=7, label=d)
               for d, c in sorted(seen.items())]
    ax.legend(handles=handles, loc='upper left', fontsize=6,
              framealpha=0.3, facecolor='#1a1208',
              edgecolor='#3a2c10', labelcolor='#f0e6cc')
    plt.tight_layout()
    plt.savefig("tsne_territories.png", dpi=150,
                bbox_inches='tight', facecolor='#1a1208')
    plt.close()
    print("✅ Saved: tsne_territories.png")


def main():
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.txt")):
        print(f"ERROR: No files in '{DATA_DIR}/'. Run clean_data.py first.")
        return

    print("Step 1: Building territory chunks...")
    chunks = parse_clean_files(DATA_DIR)
    tok_counts   = [count_tokens(c["text"]) for c in chunks]
    topic_counts = Counter(c["topic"] for c in chunks)
    print(f"  Total    : {len(chunks)}")
    print(f"  Anchors  : {sum(1 for c in chunks if c['is_anchor'])}")
    print(f"  Avg tok  : {int(sum(tok_counts)/len(tok_counts))}")
    print(f"  Max tok  : {max(tok_counts)}")
    for topic, cnt in topic_counts.most_common():
        print(f"  {topic:12}: {cnt}")

    print("\nStep 2: Embedding + storing in FAISS...")
    index, embeddings = embed_and_store(chunks)

    print("\nStep 3: t-SNE visualization...")
    visualize_tsne(chunks, embeddings)

    print(f"\n✅ Done.")
    print(f"   Chunks  : {len(chunks)}")
    print(f"   Index   : {INDEX_FILE}")
    print(f"   Meta    : {META_FILE}")
    print(f"   t-SNE   : tsne_territories.png")
    print("   Next    : run chatbot.py")


if __name__ == "__main__":
    main()