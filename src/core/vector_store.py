from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
from src.core.chunking import text_chunking
from src.core.embeding import embed_model
from src.core.storing import Milvus
from src.utils.indexing_util import index_field_data

class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 1000,
        index_type: str = "HNSW"
    ):
        """Initialize the vector store with document processing and storage components.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            batch_size: Size of batches for vector database operations
            index_type: Type of vector index to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.index_type = index_type
        
        # Initialize components
        self.embedder = embed_model.huggingface_embeding(embedding_model_name)
        self.db = Milvus(batch_size=batch_size)
        
        # Ensure collection exists with proper schema
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize vector database collection with proper schema."""
        self.db.create_schema(
            Schema_details=index_field_data,
            index_type=self.index_type,
            Drop_collection=False
        )
        self.db.load_collection()
    
    def add_documents(
        self,
        documents: Union[str, List[str], Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> bool:
        """Process and add documents to the vector store.
        
        Args:
            documents: Documents to process. Can be a single string, list of strings,
                     or a Path to a text file
            chunk_size: Optional override for chunk size
            chunk_overlap: Optional override for chunk overlap
            batch_size: Optional override for batch size
            
        Returns:
            bool: True if documents were added successfully
        """
        # Handle different input types
        if isinstance(documents, Path) or (isinstance(documents, str) and os.path.exists(documents)):
            with open(documents, 'r', encoding='utf-8') as f:
                text = f.read()
            documents = [text]
        elif isinstance(documents, str):
            documents = [documents]
            
        # Use provided parameters or defaults
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        batch_size = batch_size or self.batch_size
        
        # Process all documents
        all_chunks = []
        for doc in documents:
            chunks = text_chunking.recursive_text_splitter(
                doc,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            all_chunks.extend(chunks)
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(all_chunks)
        
        # Store in vector database
        return self.db.insert_data(all_chunks, embeddings, batch_size=batch_size)
    
    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        use_ann: bool = True,
        filter_expr: Optional[str] = None,
        **search_params
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            use_ann: Whether to use approximate nearest neighbor search
            filter_expr: Optional filter expression
            **search_params: Additional search parameters
            
        Returns:
            List of documents with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Perform search
        results = self.db.search(
            query_embedding=query_embedding,
            limit=limit,
            use_ann=use_ann,
            filter_expr=filter_expr,
            **search_params
        )
        
        return results
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None
    ) -> bool:
        """Add texts with optional metadata directly without chunking.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts for each text
            batch_size: Optional override for batch size
            
        Returns:
            bool: True if texts were added successfully
        """
        # Generate embeddings
        embeddings = self.embedder.embed_documents(texts)
        
        # Store in vector database
        return self.db.insert_data(texts, embeddings, batch_size=batch_size)
    
    def hybrid_search(
        self,
        query: str,
        filter_expr: Optional[str] = None,
        limit: int = 5,
        **search_params
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity with metadata filtering.
        
        Args:
            query: The search query
            filter_expr: Expression for filtering results
            limit: Maximum number of results
            **search_params: Additional search parameters
            
        Returns:
            List of documents with similarity scores
        """
        return self.similarity_search(
            query=query,
            limit=limit,
            filter_expr=filter_expr,
            use_ann=True,  # Always use ANN for hybrid search
            **search_params
        )