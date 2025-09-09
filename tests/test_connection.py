from pymilvus import MilvusClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    try:
        logger.info("Testing Milvus connection...")
        client = MilvusClient("http://localhost:19530")
        databases = client.list_databases()
        logger.info(f"Connected successfully! Available databases: {databases}")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()