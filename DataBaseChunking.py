from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_DIR       = Path("data_structured")
CHROMA_DIR     = "chroma_db"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 100
CHUNK_MIN_SIZE = 150
PREVIEW_CHUNKS = 5


def load_documents() -> list[Document]:
    docs = []
    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")
        dynasty = txt_file.stem.replace("_", " ").title()
        docs.append(Document(
            page_content=content,
            metadata={"source": txt_file.name, "dynasty": dynasty}
        ))
        print(f"  Loaded: {txt_file.name} ({len(content)} chars)")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nPHARAOH:",
            "\n\n",
            "\n",
            " ",
        ]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= CHUNK_MIN_SIZE]
    return chunks


def preview_chunks(chunks: list[Document]):
    print(f"\n{'='*60}")
    print(f"CHUNK PREVIEW - showing first {PREVIEW_CHUNKS} of {len(chunks)} chunks")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(chunks[:PREVIEW_CHUNKS]):
        print(f"-- Chunk {i+1} --")
        print(f"Source   : {chunk.metadata.get('source')}")
        print(f"Dynasty  : {chunk.metadata.get('dynasty')}")
        print(f"Chars    : {len(chunk.page_content)}")
        print(f"Content  :\n{chunk.page_content}")
        print()

    sizes = [len(c.page_content) for c in chunks]
    print(f"-- Size Stats --")
    print(f"  Min  : {min(sizes)} chars")
    print(f"  Max  : {max(sizes)} chars")
    print(f"  Avg  : {int(sum(sizes)/len(sizes))} chars")
    print(f"  Total chunks: {len(chunks)}")
    print()


def store_in_chroma(chunks: list[Document]):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("Embedding chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print(f"Saved to '{CHROMA_DIR}/'")
    return vectorstore


def test_retrieval(vectorstore):
    test_queries = [
        "Who was Khufu?",
        "What did Ramesses II achieve?",
        "Tell me about Cleopatra",
        "Which pharaoh built the Great Pyramid?",
    ]

    print(f"\n{'='*60}")
    print("RETRIEVAL TEST")
    print(f"{'='*60}\n")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    for query in test_queries:
        print(f"Q: {query}")
        results = retriever.invoke(query)
        for r in results:
            snippet = r.page_content[:200].replace("\n", " ")
            print(f"  -> [{r.metadata.get('source')}] {snippet}...")
        print()


def main():
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.txt")):
        print(f"ERROR: No .txt files found in '{DATA_DIR}/'. Run clean_data.py first.")
        return

    print("Step 1: Loading dynasty files...")
    docs = load_documents()
    print(f"  -> {len(docs)} files loaded\n")

    print("Step 2: Chunking...")
    chunks = chunk_documents(docs)
    print(f"  -> {len(chunks)} chunks created\n")

    print("Step 3: Previewing chunks...")
    preview_chunks(chunks)

    input("Chunks look good? Press Enter to embed and store, or Ctrl+C to abort: ")

    print("\nStep 4: Embedding + storing in ChromaDB...")
    vectorstore = store_in_chroma(chunks)

    print("\nStep 5: Testing retrieval...")
    test_retrieval(vectorstore)

    print("\nDone! ChromaDB is ready.")


if __name__ == "__main__":
    main()