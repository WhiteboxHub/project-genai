from pymilvus import MilvusClient, DataType
import os
from dotenv import load_dotenv
from src.utils.logger import logger
from src.utils.indexing_util import milvus_indexes, index_field_data
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
load_dotenv()
import uuid

# Generate a random UUID (version 4)

class chromaDB:
    def store_data():
        pass
    def retrive_data():
        pass


class Milvus:
    def __init__(self, batch_size: int = 1000):
        """Initialize Milvus client with configuration.
        
        Args:
            batch_size: Size of batches for data insertion
        """
        self.url = os.getenv("MILVUS_URL", "http://localhost:19530")
        self.client = MilvusClient(
            uri=self.url,
            token=os.getenv("MILVUS_TOKEN", ""),  # Support token-based auth
            db_name=os.getenv("DB_NAME", "default")
        )
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.batch_size = batch_size
        self._ensure_database()

    @logger
    def _ensure_database(self):
        """Ensure the configured database exists."""
        try:
            if self.db_name not in self.list_db():
                self.create_database(self.db_name)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure database exists: {e}")

    @logger
    def create_database(self, db_name: str):
        """Create a new database if it doesn't exist."""
        try:
            if db_name not in self.list_db():
                self.client.create_database(db_name=db_name)
                print(f"Created database: {db_name}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create database: {e}")

    @logger
    def list_db(self) -> List[str]:
        """List all available databases."""
        return self.client.list_databases()

    @logger
    def describe_db(self, db_name: str) -> Dict[str, Any]:
        """Get detailed information about a database."""
        try:
            return self.client.describe_database(db_name=db_name)
        except Exception as e:
            raise RuntimeError(f"Failed to describe database: {e}")

    @logger
    def create_schema(self, Schema_details: List[Dict], index_type: str, Drop_collection: bool = False) -> bool:
        """Create a collection schema with specified fields and index.
        
        Args:
            Schema_details: List of field definitions
            index_type: Type of index to create (e.g., "HNSW", "IVF_FLAT")
            Drop_collection: Whether to drop existing collection
        """
        if not Schema_details:
            raise ValueError("Schema details cannot be empty")
            
        # Validate index type
        if index_type not in milvus_indexes:
            raise ValueError(f"Unsupported index type: {index_type}. Supported types: {list(milvus_indexes.keys())}")

        try:
            # Check if collection exists
            if self.collection_name in self.client.list_collections():
                if Drop_collection:
                    print(f"Dropping existing collection '{self.collection_name}'...")
                    self.client.drop_collection(self.collection_name)
                else:
                    print(f"Collection '{self.collection_name}' already exists. Use Drop_collection=True to recreate.")
                    return False

            # Create schema
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )

            # Add fields to schema
            for field in Schema_details:
                schema.add_field(**field)

            # Prepare index
            index_data = milvus_indexes[index_type]
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",  # Using consistent field name
                metric_type=index_data['metric_type'],
                index_type=index_data['index_type'],
                index_name=index_data['index_name'],
                params=index_data['params']
            )

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            print(f"Successfully created collection '{self.collection_name}' with index type '{index_type}'")
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")

    @logger
    def load_collection(self, timeout: int = 60):
        """Load collection into memory with timeout."""
        try:
            start_time = time.time()
            self.client.load_collection(self.collection_name)
            
            # Wait for collection to be loaded
            while time.time() - start_time < timeout:
                if self.client.describe_collection(self.collection_name).get('loaded'):
                    print(f"Collection '{self.collection_name}' loaded successfully")
                    return True
                time.sleep(1)
            
            raise TimeoutError(f"Timeout waiting for collection to load after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to load collection: {e}")

    @logger
    def insert_data(self, text_chunks: List[str], chunk_embeddings: List[List[float]], batch_size: Optional[int] = None):
        """Insert data in batches with progress tracking.
        
        Args:
            text_chunks: List of text chunks to insert
            chunk_embeddings: List of embeddings corresponding to text chunks
            batch_size: Optional override for batch size
        """
        if len(text_chunks) != len(chunk_embeddings):
            raise ValueError("Number of text chunks must match number of embeddings")

        batch_size = batch_size or self.batch_size
        total_chunks = len(text_chunks)
        inserted_count = 0
        
        try:
            # Process in batches
            for i in tqdm(range(0, total_chunks, batch_size), desc="Inserting data batches"):
                batch_texts = text_chunks[i:i + batch_size]
                batch_embeddings = chunk_embeddings[i:i + batch_size]
                
                # Prepare batch data
                batch_data = [
                    {
                        "pk": str(uuid.uuid4()),
                        "embedding": emb,
                        "text": text
                    }
                    for text, emb in zip(batch_texts, batch_embeddings)
                ]
                
                # Insert batch
                result = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                
                inserted_count += len(batch_data)
                
            # Ensure data is persisted
            self.client.flush(self.collection_name)
            print(f"Successfully inserted {inserted_count} records")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to insert data: {e}")

    @logger
    def search_similar(self, query_embedding: list, limit: int = 5, output_fields: list = None):
        """Search for similar vectors using basic similarity search."""
        if output_fields is None:
            output_fields = ["text"]
            
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                field_name="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")

    @logger
    def ann_search(self, query_embedding: list, limit: int = 5, index_type: str = "HNSW", 
                  ef_search: int = 100, output_fields: list = None):
        """Search using approximate nearest neighbors (ANN)."""
        if output_fields is None:
            output_fields = ["text"]
            
        try:
            # Get index configuration
            if index_type not in milvus_indexes:
                raise ValueError(f"Unsupported index type: {index_type}")
                
            index_config = milvus_indexes[index_type]
            search_params = {
                "metric_type": index_config["metric_type"],
                "params": index_config["params"].copy()
            }
            
            # Add ef_search for HNSW-based indexes
            if index_type.startswith("HNSW"):
                search_params["params"]["ef"] = ef_search
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                field_name="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to perform ANN search: {e}")

    @logger
    def search(self, query_text: str = None, query_embedding: list = None, 
               filter_expr: str = None, limit: int = 5, output_fields: list = None, 
               use_ann: bool = True, **ann_params):
        """Unified search interface supporting both text and vector search with filtering."""
        if output_fields is None:
            output_fields = ["text"]
            
        try:
            # Handle text query if provided
            if query_text and not query_embedding:
                raise ValueError("Text search requires embedding model to be configured")
                
            search_vec = query_embedding
            
            if use_ann:
                results = self.ann_search(
                    query_embedding=search_vec,
                    limit=limit,
                    output_fields=output_fields,
                    **ann_params
                )
            else:
                results = self.search_similar(
                    query_embedding=search_vec,
                    limit=limit,
                    output_fields=output_fields
                )
            
            # Apply filtering if specified
            if filter_expr:
                filtered_results = self.client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=output_fields
                )
                
                # Combine search results with filtering
                final_results = [r for r in results if r['id'] in [fr['id'] for fr in filtered_results]]
                return final_results
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to perform search: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.client.close()
        except:
            pass


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