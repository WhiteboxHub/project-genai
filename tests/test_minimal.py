from pymilvus import connections, utility
import sys
import time

def test_minimal():
    print("Starting minimal connection test...", flush=True)
    time.sleep(1)  # Small delay
    
    try:
        # Connect using lower-level API
        print("\nConnecting to Milvus...", flush=True)
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        time.sleep(1)
        
        print("\nChecking connection...", flush=True)
        print(f"Is connected: {connections.has_connection('default')}", flush=True)
        time.sleep(1)
        
        print("\nListing collections...", flush=True)
        existing_collections = utility.list_collections()
        print(f"Found collections: {existing_collections}", flush=True)
        time.sleep(1)
        
        print("\nTest completed successfully!", flush=True)
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    success = test_minimal()
    if not success:
        sys.exit(1)