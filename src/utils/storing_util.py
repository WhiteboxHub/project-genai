# src/utils/storing_util.py
from pymilvus import connections, utility

def setup_database():
    """
    Setup Milvus database - checks if database exists before creating
    """
    try:
        connections.connect(host='localhost', port='19530')
        
        # Check if database exists first
        databases = utility.list_database()
        if "testdb" in databases:
            print("Database 'testdb' already exists - using it")
        else:
            utility.create_database("testdb")
            print("Database 'testdb' created successfully")
        
        # Set this database as current
        utility.using_database("testdb")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

def get_milvus_connection():
    """
    Get or create Milvus connection
    """
    try:
        connections.connect(host='localhost', port='19530')
        return True
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return False