from pymilvus import (
    MilvusClient,
    DataType,
    Collection,
    CollectionSchema,
    FieldSchema,
    connections
)
import numpy as np
import sys
import time

def validate_connection(client):
    """Validate that we can connect to Milvus"""
    try:
        collections = client.list_collections()
        print(f"Successfully connected to Milvus. Found collections: {collections}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def create_test_collection():
    print("\nStarting Milvus test...")
    
    # 1. Connect and validate connection
    print("\nConnecting to Milvus...")
    client = MilvusClient("http://localhost:19530")
    if not validate_connection(client):
        raise Exception("Could not connect to Milvus server")
    
    # 2. Create schema fields
    print("\nCreating schema...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2)
    ]
    schema = CollectionSchema(fields=fields)
    
    # 3. Create collection
    collection_name = "test_collection"
    try:
        if collection_name in client.list_collections():
            print(f"\nDropping existing collection '{collection_name}'...")
            client.drop_collection(collection_name)
            time.sleep(2)  # Give it time to drop
        
        print(f"\nCreating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            schema=schema
        )
        time.sleep(2)  # Give it time to create
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise
    
    # 4. Insert data
    try:
        print("\nPreparing test data...")
        vectors = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        entities = [
            {"id": i, "vector": v} 
            for i, v in enumerate(vectors)
        ]
        
        print("Inserting data...")
        insert_result = client.insert(collection_name, entities)
        print(f"Insert result: {insert_result}")
        
        print("Flushing data...")
        client.flush(collection_name)
        print("Data inserted and flushed successfully")
    except Exception as e:
        print(f"Error inserting data: {e}")
        raise
    
    # 5. Create index
    try:
        print("\nCreating index...")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="L2",
            index_type="IVF_FLAT",
            index_name="vector_idx",
            params={"nlist": 128}
        )
        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        print("Index created successfully")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise
    
    # 6. Load collection
    try:
        print("\nLoading collection...")
        client.load_collection(collection_name)
        print("Collection loaded successfully")
    except Exception as e:
        print(f"Error loading collection: {e}")
        raise
    
    # 7. Search
    try:
        print("\nPerforming search...")
        search_params = {
            "anns_field": "vector",
            "param": {"nprobe": 10},
            "metric_type": "L2"
        }
        search_vector = [1.0, 2.0]
        results = client.search(
            collection_name=collection_name,
            data=[search_vector],
            limit=2,
            output_fields=["id"],
            **search_params
        )
        
        print(f"\nSearch results: {results}")
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during search: {e}")
        raise

if __name__ == "__main__":
    try:
        create_test_collection()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)