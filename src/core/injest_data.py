import os
from chunking import FileChunking
from embeding import EmbedModel
from storing import MilvusDB
from text_extraction import readfile


def file_already_ingested(db: MilvusDB, filename: str) -> bool:
    """
    Check if a file has already been ingested into Milvus.
    Queries the COSINE collection for filename.
    """
    # Access the internal collection for COSINE similarity
    results = db.cosine_collection.query(
        expr=f'filename == "{filename}"',
        output_fields=["filename"],
        limit=1
    )
    return len(results) > 0


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../src/core
    PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))    # .../project-genai
    DATA_DIR = os.path.join(PROJECT_DIR, "Data")                # .../project-genai/data

    print("Looking for PDFs in:", DATA_DIR)

    # --- Initialize Milvus ---
    print("Initializing Milvus...")
    db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())

    # --- Loop over all PDFs in data folder ---
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        print(f"\n📄 Processing file: {filename}")

        # --- Deduplication check ---
        if file_already_ingested(db, filename):
            print(f"⚠️ Skipping {filename} (already ingested)")
            continue

        # --- Read file ---
        text = readfile(file_path).strip()
        print(f"Extracted {len(text)} characters")

        if not text:
            print(f"⚠️ Skipped empty file: {filename}")
            continue

        # --- Chunk text ---
        chunks = FileChunking.sentence(text)
        print(f"Created {len(chunks)} chunks")

        # --- Metadata (filename repeated for each chunk) ---
        filenames = [filename] * len(chunks)

        # --- Store in Milvus ---
        print("📥 Storing data in Milvus...")
        inserted_info = db.store_data(chunks, filenames)
        print(f"✅ Stored {inserted_info['inserted']} chunks from {filename}")

# --- Optional: test retrieval ---

    test_query = "What is Agentic AI?"  # you can change this
    print(f"\n🔎 Testing retrieval for query: {test_query}")

    # COSINE search
    results_cos = db.ann_cosine_search(test_query, k=3)
    print("\n--- COSINE Search Results ---")
    for r in results_cos:
        print(f"[{r['filename']}] (score={r['score']:.4f}) -> {r['text'][:120]}...")

    # IP search
    results_ip = db.ip_search(test_query, k=3)
    print("\n--- IP Search Results ---")
    for r in results_ip:
        print(f"[{r['filename']}] (score={r['score']:.4f}) -> {r['text'][:120]}...")
#==============OUTPUT====================
# ✅ Connected to Milvus at localhost:19530

# 📄 Processing file: langchain.pdf
# ⚠️ Skipping langchain.pdf (already ingested)

# 📄 Processing file: agentic-ai.pdf
# ⚠️ Skipping agentic-ai.pdf (already ingested)

# 📄 Processing file: chunking_strategies.pdf
# ⚠️ Skipping chunking_strategies.pdf (already ingested)

# 🔎 Testing retrieval for query: What is Agentic AI?

# --- COSINE Search Results ---
# [agentic-ai.pdf] (score=1.0000) -> What is Agentic AI?...
# [agentic-ai.pdf] (score=1.0000) -> What is agentic AI?...
# [agentic-ai.pdf] (score=0.8518) -> Agentic AI generally refers to AI systems that 
# possess the capacity to make autonomous 
# decisions and take actions to a...

# --- IP Search Results ---
# [agentic-ai.pdf] (score=1.0000) -> What is Agentic AI?...
# [agentic-ai.pdf] (score=1.0000) -> What is agentic AI?...
# [agentic-ai.pdf] (score=0.8518) -> Agentic AI generally refers to AI systems that 
# possess the capacity to make autonomous 
# decisions and take actions to a...        