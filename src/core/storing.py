#from milvus import MilvusClient
from pymilvus import Index, Collection, FieldSchema, CollectionSchema, DataType, connections, utility
from sentence_transformers import SentenceTransformer
import numpy as np

class chromaDB: 
    def store_data(): #store the embedding and text
        pass
    def retrive_data():
        pass


    #---------------Milvus-start-------------------------

class Milvus:
    @staticmethod
    def Milvus_store_data_insert_and_query(
        collection_name: str = "my_collection",
        vector_dim: int = 384,
        texts: list[str] = None,
        query_text: str = None,
        top_k: int = 10
    ):

        """
        Create collection if it doesn't exist, insert embeddings, and query top-k similar vectors.
        """
        if texts is None:
            texts = []

        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Check if collection exists
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            print(f"Collection '{collection_name}' exists, using existing collection.")
        else:
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
            ]
            schema = CollectionSchema(fields, description="Collection for text embeddings")
            collection = Collection(name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created successfully.")

        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Insert embeddings if texts are provided
        if texts:
            embeddings = model.encode(texts).tolist()
            collection.insert([embeddings])
            collection.flush()
            print(f"Inserted {len(texts)} records into '{collection_name}'.")

        # Query if query_text is provided
        top_results = []
        if query_text:
            query_embedding = model.encode([query_text]).tolist()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # collection is your Milvus collection object
            index_params = {
                "index_type": "IVF_FLAT",  # can also be HNSW, IVF_SQ8, etc.
                "metric_type": "L2",
                "params": {"nlist": 128}   # number of clusters
            }

            index = Index(collection, "embedding", index_params=index_params)

            collection = Collection(collection_name)
            collection.load()  # Load collection into memory

            results = collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=None
            )
            for hits in results:
                for hit in hits:
                    top_results.append((hit.id, hit.distance))

        return top_results


# Example usage
if __name__ == "__main__":
    sample_texts = [
        "Retrieval-Augmented Generation is a method combining retrieval and generation.",
        "Sentence chunking keeps full sentences intact."
    ]
    query = "How does retrieval-augmented generation work?"

    top_matches = Milvus.Milvus_store_data_insert_and_query(
        collection_name="my_collection",
        vector_dim=384,
        texts=sample_texts,
        query_text=query,
        top_k=10
    )

    print("Top matches (ID, distance):")
    for item in top_matches:
        print(item)

    
    #---------------Milvus-End-------------------------
    
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


  #   -----------     Output      -----------
"""   
 (.venv) hema@Mac project-genai % /Users/hema/Desktop/project-genai/.venv/bin/python /Users/hema/Desktop/proj
ect-genai/src/core/storing.py
Collection 'my_collection' exists, using existing collection.
Inserted 2 records into 'my_collection'.
Top matches (ID, distance):
(460663754245597922, 0.2971716821193695)
(460663754245597925, 0.2971716821193695)
(460663754245597928, 0.2971716821193695)
(.venv) hema@Mac project-genai % 

"""