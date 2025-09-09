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
#     query = "What is in the document?"
#     print(f"\n🔎 Testing retrieval for query: {query}")
#     results = db.search(query, k=3)
#     for r in results:
#         print("Retrieved:", r)
#   #========OUTPUT==========================
  #🔎 Testing retrieval for query: What is in the document?
#Retrieved: {'filename': 'chunking_strategies.pdf', 'text': 'A template consists of a core, which contains constant information, and slots, where variable \ninformation can be stored.', 'score': 0.4037086069583893}
#Retrieved: {'filename': 'chunking_strategies.pdf', 'text': 'A template consists of a core, which contains constant information, and slots, where variable \ninformation can be stored.', 'score': 0.4037086069583893}
#Retrieved: {'filename': 'langchain.pdf', 'text': 'PDFDataExtractor: A Tool for \nReading Scientific Text and Interpreting Metadata from the Typeset \nLiterature in the Portable Document For mat.', 'score': 0.37489455938339233}      