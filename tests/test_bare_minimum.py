from pymilvus import MilvusClient
import sys

print("Starting bare minimum test...")

try:
    # Connect
    print("\nConnecting to Milvus...")
    client = MilvusClient("http://localhost:19530")
    
    # List collections
    print("\nListing collections...")
    collections = client.list_collections()
    print(f"Found collections: {collections}")
    
    # Create collection
    collection_name = "test_collection"
    if collection_name in collections:
        print(f"\nDropping collection '{collection_name}'...")
        client.drop_collection(collection_name)
    
    print(f"\nCreating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        dimension=2
    )
    
    # Insert data
    print("\nInserting data...")
    data = [
        {"vector": [1.0, 2.0]},
        {"vector": [3.0, 4.0]}
    ]
    result = client.insert(collection_name, data)
    print(f"Insert result: {result}")
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)