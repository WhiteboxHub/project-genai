from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()
class chromaDB:
   class ChromaDB:
    @staticmethod
    def store_data( chunks, embedding_model=None):
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        documents = [
            Document(page_content=chunk, metadata={"chunk_id": idx})
            for idx, chunk in enumerate(chunks)
        ]

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory="chroma_db1",
            collection_name="sample"
        )

        return vector_store
    
    def retrieve_data(query, model, collection, top_k=3):
     query_embedding = model.embed_query(query)
     results = collection.similarity_search_by_vector(query_embedding, k=top_k)
     print(f"\nTop {top_k} results for query: {query}")
     for idx, doc in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(doc.page_content)
        print("-" * 50)

     return results

class Milvus:
    def store_data():
        pass
    def retrive_data():
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