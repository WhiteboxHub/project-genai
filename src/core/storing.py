import numpy as np
from typing import List, Dict, Any

class ChromaDB:
    @staticmethod
    def store_data(embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """
        Store embeddings and texts in ChromaDB
        """
        try:
            import chromadb
            client = chromadb.Client()
            collection = client.create_collection("documents")
            
            ids = [f"doc_{i}" for i in range(len(texts))]
            
            if metadata is None:
                metadata = [{"text": text} for text in texts]
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            print("✓ Data stored in ChromaDB")
            return True
        except Exception as e:
            print(f"✗ Error storing in ChromaDB: {e}")
            return False

    @staticmethod
    def retrieve_data(query_embedding: np.ndarray, n_results: int = 5):
        """
        Retrieve similar documents from ChromaDB
        """
        try:
            import chromadb
            client = chromadb.Client()
            collection = client.get_collection("documents")
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"✗ Error retrieving from ChromaDB: {e}")
            return None

class Milvus:
    @staticmethod
    def store_data(embeddings: np.ndarray, texts: List[str]):
        """
        Store embeddings and texts in Milvus
        """
        try:
            from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
            import json
            
            # Connect to Milvus
            connections.connect("default", host="localhost", port="19530")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
            ]
            schema = CollectionSchema(fields, "document_collection")
            
            # Create collection
            collection = Collection("documents", schema)
            
            # Insert data
            entities = [embeddings.tolist(), texts]
            collection.insert(entities)
            
            print("✓ Data stored in Milvus")
            return True
        except Exception as e:
            print(f"✗ Error storing in Milvus: {e}")
            return False

    @staticmethod
    def retrieve_data(query_embedding: np.ndarray, n_results: int = 5):
        """
        Retrieve similar documents from Milvus
        """
        try:
            from pymilvus import connections, Collection
            
            connections.connect("default", host="localhost", port="19530")
            collection = Collection("documents")
            collection.load()
            
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=n_results,
                output_fields=["text"]
            )
            return results
        except Exception as e:
            print(f"✗ Error retrieving from Milvus: {e}")
            return None

class PGVectorDB:
    @staticmethod
    def store_data(embeddings: np.ndarray, texts: List[str]):
        """
        Store embeddings and texts in PostgreSQL with pgvector
        """
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            
            conn = psycopg2.connect(
                dbname="vectordb",
                user="postgres",
                password="password",
                host="localhost"
            )
            cur = conn.cursor()
            
            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    embedding VECTOR(384),
                    text TEXT
                )
            """)
            
            # Insert data
            data = [(emb.tolist(), text) for emb, text in zip(embeddings, texts)]
            execute_values(
                cur,
                "INSERT INTO documents (embedding, text) VALUES %s",
                data
            )
            
            conn.commit()
            print("✓ Data stored in PGVectorDB")
            return True
        except Exception as e:
            print(f"✗ Error storing in PGVectorDB: {e}")
            return False

    @staticmethod
    def retrieve_data(query_embedding: np.ndarray, n_results: int = 5):
        """
        Retrieve similar documents from PostgreSQL with pgvector
        """
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                dbname="vectordb",
                user="postgres",
                password="password",
                host="localhost"
            )
            cur = conn.cursor()
            
            cur.execute("""
                SELECT text, embedding <=> %s as distance
                FROM documents
                ORDER BY distance
                LIMIT %s
            """, (query_embedding.tolist(), n_results))
            
            results = cur.fetchall()
            return results
        except Exception as e:
            print(f"✗ Error retrieving from PGVectorDB: {e}")
            return None

class PineconeDB:
    @staticmethod
    def store_data(embeddings: np.ndarray, texts: List[str]):
        """
        Store embeddings and texts in Pinecone
        """
        try:
            import pinecone
            
            pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
            
            # Create index if not exists
            if "documents" not in pinecone.list_indexes():
                pinecone.create_index("documents", dimension=embeddings.shape[1])
            
            index = pinecone.Index("documents")
            
            # Prepare data for upsert
            vectors = []
            for i, (emb, text) in enumerate(zip(embeddings, texts)):
                vectors.append((f"doc_{i}", emb.tolist(), {"text": text}))
            
            index.upsert(vectors=vectors)
            print("✓ Data stored in Pinecone")
            return True
        except Exception as e:
            print(f"✗ Error storing in Pinecone: {e}")
            return False

    @staticmethod
    def retrieve_data(query_embedding: np.ndarray, n_results: int = 5):
        """
        Retrieve similar documents from Pinecone
        """
        try:
            import pinecone
            
            pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
            index = pinecone.Index("documents")
            
            results = index.query(
                vector=query_embedding.tolist(),
                top_k=n_results,
                include_metadata=True
            )
            return results
        except Exception as e:
            print(f"✗ Error retrieving from Pinecone: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Sample data
    embeddings = np.random.rand(10, 384)  # 10 embeddings of dimension 384
    texts = [f"Sample text {i}" for i in range(10)]
    
    # Test ChromaDB
    ChromaDB.store_data(embeddings, texts)
    results = ChromaDB.retrieve_data(embeddings[0])
    print("ChromaDB results:", results)