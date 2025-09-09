
import os
from chunking import FileChunking
from embeding import EmbedModel
from storing import MilvusDB
from text_extraction import readfile


def file_already_ingested(db: MilvusDB, filename: str) -> bool:
    """
    Check if a file has already been ingested into Milvus.
    We just search by filename and see if there are any results.
    """
    results = db.collection.query(
        expr=f'filename == "{filename}"',
        output_fields=["filename"],
        limit=1
    )
    return len(results) > 0


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../src/core
    PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))    # .../project-genai
    DATA_DIR = os.path.join(PROJECT_DIR, "data")                # .../project-genai/data

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
    query = "What is in the document?"
    print(f"\n🔎 Testing retrieval for query: {query}")
    results = db.search(query, k=3)
    for r in results:
        print("Retrieved:", r)
  