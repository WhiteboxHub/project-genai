from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import sys

print("Starting simple test...", flush=True)

try:
    print("Creating client...", flush=True)
    client = MilvusClient(
        uri="http://localhost:19530",
        token="",  # No authentication
        db_name="default"  # Explicit database name
    )
    
    print("Listing collections...", flush=True)
    collections = client.list_collections()
    print(f"Collections: {collections}", flush=True)
    
    # Create a simple collection
    collection_name = "simple_test"
    if collection_name in collections:
        print(f"\nDropping existing collection '{collection_name}'...")
        client.drop_collection(collection_name)
    
    print(f"\nCreating collection '{collection_name}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100)
    ]
    schema = CollectionSchema(fields=fields)
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    print("Collection created successfully")
    
    # Insert simple test data
    print("\nInserting test data...")
    test_data = [{
        "id": "1",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "text": "test vector 1"
    }]
    
    client.insert(collection_name, test_data)
    print("Data inserted successfully")
    
    print("Test completed successfully!", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)