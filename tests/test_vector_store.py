from src.core.vector_store import VectorStore
import logging
from pathlib import Path
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_store():
    """Test the vector store functionality with sample data."""
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            embedding_model_name="BAAI/bge-small-en-v1.5",
            chunk_size=300,
            chunk_overlap=30,
            batch_size=500,
            index_type="HNSW"
        )
        
        # Test adding a single document
        logger.info("\nTesting single document addition...")
        sample_text = """
        Vector databases are specialized database systems designed to store and query vector embeddings efficiently.
        They are crucial for machine learning applications, particularly in natural language processing and computer vision.
        Modern vector databases like Milvus use sophisticated indexing methods such as HNSW (Hierarchical Navigable Small World)
        to enable fast approximate nearest neighbor search in high-dimensional spaces.
        """
        
        success = vector_store.add_documents(sample_text)
        logger.info(f"Single document addition {'successful' if success else 'failed'}")
        
        # Test adding multiple documents
        logger.info("\nTesting multiple document addition...")
        data_dir = Path("Data")
        if data_dir.exists():
            pdf_files = [str(f) for f in data_dir.glob("*.pdf")]
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF files")
                for pdf_file in pdf_files:
                    logger.info(f"Processing {pdf_file}...")
                    success = vector_store.add_documents(pdf_file)
                    logger.info(f"Document addition {'successful' if success else 'failed'}")
        
        # Test similarity search
        logger.info("\nTesting similarity search...")
        query = "What are vector databases used for?"
        results = vector_store.similarity_search(
            query=query,
            limit=3,
            use_ann=True
        )
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Text: {result.get('text', '')[:200]}...")
            logger.info(f"Score: {result.get('score', 'N/A')}")
        
        # Test hybrid search with filtering
        logger.info("\nTesting hybrid search...")
        query = "machine learning applications"
        filter_expr = "text like '%database%'"
        results = vector_store.hybrid_search(
            query=query,
            filter_expr=filter_expr,
            limit=3
        )
        logger.info(f"Hybrid search results for '{query}' with filter '{filter_expr}':")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Text: {result.get('text', '')[:200]}...")
            logger.info(f"Score: {result.get('score', 'N/A')}")
        
        logger.info("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    test_vector_store()