from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pinecone


class ChromaDB:
    def __init__(self, collection_name="genai_collection"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.Client()
        self.collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def store_data(self, ids, texts, metadatas=None):
        embeddings = self.embedding_model.encode(texts).tolist()
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[ChromaDB] Stored {len(texts)} documents.")

    def retrieve_data(self, query, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results


class PineconeDB:
    def __init__(self, api_key, index_name="genai-index", dimension=384):
        pinecone.init(api_key=api_key)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        self.index = pinecone.Index(index_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def store_data(self, ids, texts, metadatas=None):
        embeddings = self.embedding_model.encode(texts).tolist()
        vectors = []
        for i, emb in enumerate(embeddings):
            vector = {
                "id": ids[i],
                "values": emb,
                "metadata": metadatas[i] if metadatas else {"text": texts[i]}
            }
            vectors.append(vector)
        self.index.upsert(vectors=vectors)
        print(f"[PineconeDB] Stored {len(texts)} documents.")

    def retrieve_data(self, query, top_k=3):
        query_vector = self.embedding_model.encode([query]).tolist()[0]
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return results
