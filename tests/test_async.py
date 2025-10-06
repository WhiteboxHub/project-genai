from pymilvus import MilvusClient
import sys
import time

print("Starting async test...")

try:
    # Connect
    print("\nConnecting to Milvus...")
    client = MilvusClient(
        uri="http://localhost:19530",
        timeout=30  # 30 second timeout
    )
    
    # List collections
    print("\nListing collections...")
    collections = client.list_collections()
    print(f"Found collections: {collections}")
    
    # Create collection
    collection_name = "test_async"
    if collection_name in collections:
        print(f"\nDropping collection '{collection_name}'...")
        client.drop_collection(collection_name)
        time.sleep(5)  # Give it time to drop
    
    print(f"\nCreating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        dimension=2,
        primary_field_name="id",
        vector_field_name="vector"
    )
    time.sleep(5)  # Give it time to create
    
    # Verify collection exists
    print("\nVerifying collection was created...")
    collections = client.list_collections()
    if collection_name not in collections:
        raise Exception("Collection was not created successfully")
    print("Collection created successfully")
    
    # Insert data
    print("\nInserting data...")
    data = [
        {"vector": [1.0, 2.0], "id": 1},
        {"vector": [3.0, 4.0], "id": 2}
    ]
    result = client.insert(collection_name, data)
    print(f"Insert result: {result}")
    
    # Flush to ensure data is persisted
    print("\nFlushing data...")
    client.flush(collection_name)
    print("Data flushed successfully")
    
    # Get collection statistics
    print("\nGetting collection statistics...")
    stats = client.get_collection_stats(collection_name)
    print(f"Collection stats: {stats}")
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)