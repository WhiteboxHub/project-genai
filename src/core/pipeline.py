# src/core/pipeline.py
from text_extraction import readfile
from chunking import FileChunking
from embeding import EmbedModel
from storing import MilvusDB
from general import GeneralLLM

def ingest_file(file_path: str, db: MilvusDB):
    """
    Ingest a single file into Milvus.
    """
    text = readfile(file_path)
    if not text.strip():
        print(f"⚠️ File is empty: {file_path}")
        return
    print(f"File Path : {text}")
    chunks = FileChunking.sentence(text)
    print(f"Chunks Length : {len(chunks)}")
    filenames = [file_path] * len(chunks)

    result = db.store_data(chunks, filenames)
    print(f"✅ Stored {result['inserted']} chunks from {file_path}")


def query_pipeline(query: str, db: MilvusDB, llm: GeneralLLM):
    """
    Query Milvus and get answer from Groq.
    """
    retrieved_docs = db.retrieve_data(query, k=3)
    if not retrieved_docs:
        print("⚠️ No relevant documents found in Milvus.")
        return

    answer = llm.gen_ans(retrieved_docs, query)
    print("\n🤖 Groq Answer:")
    print(answer)


# if __name__ == "__main__":
#     # --- Configuration ---
#     file_path = "Decision_Trees_Random_Forests_Notes.pdf"
#     query = "Explain decision trees vs random forests"

#     # --- Initialize Milvus and LLM ---
#     db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())
#     llm = GeneralLLM()

#     # --- Step 1: Ingest the single file ---
#     ingest_file(file_path, db)

#     # --- Step 2: Query ---
#     query_pipeline(query, db, llm)
