import sys
print("Starting test script...")  # Immediate feedback

try:
    from pymilvus import MilvusClient, DataType
    print("Successfully imported pymilvus")
except Exception as e:
    print(f"Error importing pymilvus: {e}")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("Successfully imported sentence_transformers")
except Exception as e:
    print(f"Error importing sentence_transformers: {e}")
    sys.exit(1)

import logging
import os
from dotenv import load_dotenv
import uuid

print("All imports successful")

def main():
    try:
        # Basic connection test
        print("\nTesting Milvus connection...")
        client = MilvusClient("http://localhost:19530")
        collections = client.list_collections()
        print(f"Connected to Milvus. Available collections: {collections}")

        # Collection setup
        collection_name = "test_collection"
        print(f"\nSetting up collection: {collection_name}")
        
        if collection_name in collections:
            print("Dropping existing collection...")
            client.drop_collection(collection_name)
        
        # Create schema
        print("Creating schema...")
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        schema.add_field(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        schema.add_field(name="text", dtype=DataType.VARCHAR, max_length=800)
        
        # Create collection
        print("Creating collection...")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            index_name="vector_index",
            params={"M": 16, "efConstruction": 200}
        )
        
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print("Collection created successfully")
        
        # Test data
        print("\nPreparing test data...")
        text_chunks = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump"
        ]
        
        print("Loading embedding model...")
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print("Generating embeddings...")
        embeddings = [model.encode(text).tolist() for text in text_chunks]
        print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Insert data
        print("\nInserting data...")
        data = []
        for text, embedding in zip(text_chunks, embeddings):
            data.append({
                "pk": str(uuid.uuid4()),
                "embedding": embedding,
                "text": text
            })
        
        client.insert(collection_name, data)
        print("Data inserted successfully")
        
        # Load and search
        print("\nLoading collection for search...")
        client.load_collection(collection_name)
        
        query = "quick jumping animals"
        print(f"\nSearching for: '{query}'")
        query_embedding = model.encode(query).tolist()
        
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=2,
            output_fields=["text"],
            field_name="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"ef": 100}
            }
        )
        
        print("\nSearch results:")
        print(results)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()