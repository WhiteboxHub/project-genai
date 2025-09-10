# src/core/storing1.py
from pymilvus import connections, utility, FieldSchema, DataType, Collection, CollectionSchema
from embeding import embed_model
from chunking import file_chunking
from text_extraction import text_extraction 
import os

class Milvus:
    @staticmethod
    def Milvus_connection_create_collection():
        # Connect to Milvus
        connections.connect(host='localhost', port='19530')
        
        # Check existing collections
        existing_collections = utility.list_collections()
        print("Existing collections: ", existing_collections)
        
        # If collection already exists, return it
        if 'firstcollection' in existing_collections:
            print("Collection 'firstcollection' already exists")
            return 'firstcollection'
        
        # Adding fields
        id_field = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)
        embedding_field = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=384)
        content_field = FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=2048)

        # Creating schema
        schema = CollectionSchema(fields=[id_field, embedding_field, content_field])

        # Creating collection
        collection = Collection(name='firstcollection', schema=schema)

        # Index
        index_params = {
            'metric_type': 'L2',
            'index_type': 'HNSW',
            'params': {
                'M': 16,
                'efConstruction': 200
            }
        }
        collection.create_index(field_name='embeddings', index_params=index_params)
        collection.load()
        return collection.name

    @staticmethod
    def connect_milvus_load_collection(collection_name: str = 'firstcollection'):
        connections.connect(host='localhost', port='19530')
        collection = Collection(name=collection_name)
        collection.load()
        return collection

    @staticmethod
    def insert_data(collection):
        # Check if file exists
        txt_path = "./Data/sample.txt"
        if not os.path.exists(txt_path):
            print(f"File not found: {txt_path}")
            return
        
        # Extract text
        content = text_extraction.extract_text_from_txt(txt_path)
        if not content:
            print("No content extracted from file")
            return
        
        # Chunk text
        text_chunks = file_chunking.recursive(text=content)
        if not text_chunks:
            print("No chunks created")
            return
        
        # Generate embeddings
        model = embed_model.sentence_Transformer('all-MiniLM-L6-v2')
        content_embeddings = embed_model.get_embeddings(model, text_chunks)
        
        # CORRECTED: Check if embeddings are None or empty (without numpy)
        if content_embeddings is None or (hasattr(content_embeddings, 'size') and content_embeddings.size == 0):
            print("No embeddings generated")
            return
        
        # Convert numpy array to list for Milvus (works with or without numpy)
        try:
            content_embeddings_list = content_embeddings.tolist()
        except AttributeError:
            # If it's already a list, use it directly
            content_embeddings_list = content_embeddings
        
        # Prepare data for insertion
        if len(content_embeddings_list) != len(text_chunks):
            print(f"Mismatch: {len(content_embeddings_list)} embeddings vs {len(text_chunks)} chunks")
            return
        
        # Insert data
        try:
            insert_data = [content_embeddings_list, text_chunks]
            collection.insert(insert_data)
            print(f"Inserted {len(text_chunks)} chunks successfully")
        except Exception as e:
            print(f"Error inserting data: {e}")

    @staticmethod
    def search_milvus(query, collection, limit=3):
        # Generate query embedding
        model = embed_model.sentence_Transformer('all-MiniLM-L6-v2')
        query_embedding = embed_model.get_embeddings(model, [query])
        
        # CORRECTED: Check if query_embedding is None or empty
        if query_embedding is None or (hasattr(query_embedding, 'size') and query_embedding.size == 0):
            print("Failed to generate query embedding")
            return None
        
        # Convert to list for Milvus
        try:
            query_embedding_list = query_embedding.tolist()
        except AttributeError:
            query_embedding_list = query_embedding
        
        search_params = {
            'metric_type': 'L2',
            'params': {'ef': 50}
        }
        
        try:
            results = collection.search(
                data=[query_embedding_list[0]],  # Use the first embedding
                anns_field='embeddings',
                param=search_params,
                limit=limit,
                output_fields=['content']
            )
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return None

    @staticmethod
    def filter_Milvus_content_retrieval(search_results):
        if not search_results:
            return []
        
        contents = []
        for result in search_results[0]:
            content = result.entity.get('content')
            if content:
                contents.append(content)
        return contents

# Main execution
if __name__ == "__main__":
    try:
        # Create collection
        collection_name = Milvus.Milvus_connection_create_collection()
        print("Collection name: ", collection_name)

        # Connect and load collection
        collection = Milvus.connect_milvus_load_collection()

        # Insert data
        Milvus.insert_data(collection)

        # Search
        user_query = "what is photosynthesis"
        search_results = Milvus.search_milvus(user_query, collection, 3)
        
        if search_results:
            filtered_content = Milvus.filter_Milvus_content_retrieval(search_results)
            print(f"\nSearch results for '{user_query}':")
            for i, content in enumerate(filtered_content, 1):
                print(f"\nResult {i}:")
                print(content[:200] + "..." if len(content) > 200 else content)
        else:
            print("No search results found")

    except Exception as e:
        print(f"Error in main execution: {e}")

        #output
        """
        Collection name:  firstcollection
✓ Loaded model: sentence-transformers/all-MiniLM-L6-v2
✓ Generated embeddings for 12 texts
Inserted 12 chunks successfully
✓ Loaded model: sentence-transformers/all-MiniLM-L6-v2
✓ Generated embeddings for 1 texts

Search results for 'what is photosynthesis':

Result 1:
Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels. It can lead to complications such as neuropathy, retinopathy, kidney disease, and cardiovascular problems. Proper

Result 2:
Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels. It can lead to complications such as neuropathy, retinopathy, kidney disease, and cardiovascular problems. Proper

Result 3:
disease, and cardiovascular problems. Proper management includes monitoring blood glucose, following a healthy diet, and taking medications as prescribed.
"""