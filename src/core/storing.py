import chromadb
from sentence_transformers import SentenceTransformer

class chromaDB:      

 client = chromadb.PersistentClient(path="./chroma_storage")

def store_data(collection_name, texts, model_name, recreate_on_mismatch=True):
   
    # 1. Load embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts).tolist()
    current_dim = len(embeddings[0])

    # 2. Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Found collection '{collection_name}'")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"🆕 Created new collection '{collection_name}'")

    # 3. Check dimension ONLY if collection is not empty and has embeddings
    if collection.count() > 0:
        existing_peek = collection.peek(limit=1)
        if existing_peek and "embeddings" in existing_peek and len(existing_peek["embeddings"]) > 0:
            existing_dim = len(existing_peek["embeddings"][0])
            if existing_dim != current_dim:
                msg = f"⚠️ Dimension mismatch! Existing: {existing_dim}, New: {current_dim}"
                if recreate_on_mismatch:
                    print(msg + " → Recreating collection...")
                    client.delete_collection(name=collection_name)
                    collection = client.create_collection(name=collection_name)
                else:
                    raise ValueError(msg)

    # 4. Generate unique IDs
    start_idx = collection.count()
    ids = [f"id_{i}" for i in range(start_idx, start_idx + len(texts))]

    # 5. Add data
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(f"✅ Stored {len(texts)} documents in '{collection_name}'")

    return collection
    
    

def retrieve_data(collection_name, query_texts, model_name, n_results=3, metadata_filter=None):
    
    # 1. Ensure query_texts is a list
    if isinstance(query_texts, str):
        query_texts = [query_texts]

    # 2. Load embedding model
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(query_texts).tolist()

    # 3. Connect to ChromaDB collection
    collection = client.get_collection(name=collection_name)

    # 4. Query the collection
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=metadata_filter  # optional filter
    )

    return results

query = "What is a vector database?"
results = retrieve_data(
    collection_name="mpnet_collection",
    query_texts=query,
    model_name="sentence-transformers/all-mpnet-base-v2",
    n_results=2
)

print("Top Results:")
for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"- Document: {doc}")
    print(f"  Distance: {dist:.4f}")
    print(f"  Metadata: {meta}\n")
    

class Milvus:
    def store_data():
        pass
    def retrive_data():
        pass
    
class PGvectordb:
    def store_data():
        pass
    def retrive_data():
        pass


class pinecodedb:
    def store_data():
        pass
    def retrive_data():
        pass