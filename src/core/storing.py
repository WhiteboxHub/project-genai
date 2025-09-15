from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from langchain.schema import Document
import uuid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeException

from dotenv import load_dotenv
load_dotenv()
from langchain.vectorstores import Chroma


load_dotenv()
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
            persist_directory="chroma_db",
            collection_name="sample1"
        )

        return vector_store
    
    @staticmethod
    def retrieve_data(query, model, collection, top_k=3):
     query_embedding = model.embed_query(query)
     results = collection.similarity_search_by_vector(query_embedding, k=top_k)
     print(f"\nTop {top_k} results for query: {query}")
     for idx, doc in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(doc.page_content)
        print("-" * 50)

     return results
    


    @staticmethod
    def retrieve_data_mmr(query, model, collection, top_k=3, lambda_mult=0.5, fetch_k=20):
     query_embedding = model.embed_query(query)
     results = collection.max_marginal_relevance_search_by_vector(
        query_embedding,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

     print(f"\nTop {top_k} MMR (diverse) results for query: {query}")
     for idx, doc in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(doc.page_content)
        print("-" * 50)



class Pineconedb:        
    @staticmethod
    def create_index(index_name, dimension, metric="cosine", cloud="aws", region="us-east-1"):
      api_key = os.getenv("PINECONE_API_KEY")
      if not api_key:
        raise ValueError(" PINECONE_API_KEY is not set in environment variables.")

      try:
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Created index '{index_name}' with dimension {dimension}.")
        else:
            print(f"Index '{index_name}' already exists.")
      except PineconeException as e:
        print(f"Failed to create or access Pinecone: {str(e)}")



    
    @staticmethod
    def store_embeddings_pinecone(index_name: str, text_chunks: list[str], embeddings):

     api_key = os.getenv("PINECONE_API_KEY")
     if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")

     try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        vectors = []
        for text, vector in zip(text_chunks, embeddings):
            vector_id = str(uuid.uuid4())  # Or your own unique ID
            metadata = {"text": text}
            vectors.append((vector_id, vector.tolist(), metadata))

        index.upsert(vectors=vectors)
        print(f"Stored {len(vectors)} vectors in index '{index_name}'.")
        return True

     except Exception as e:
        print(f"Error upserting embeddings: {e}")
        return False

    @staticmethod
    def retrieve_data_from_pinecone(index_name: str, query: str, top_k=5, model_name="all-MiniLM-L6-v2"):

     api_key = os.getenv("PINECONE_API_KEY")
     if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")

     print(f"\n Embedding query: '{query}' using model '{model_name}'")
     model = SentenceTransformer(model_name)
     query_embedding = model.encode([query])[0].tolist()
     print(f" Query embedding generated (length: {len(query_embedding)}): {query_embedding[:5]}...")

     try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        print(f"🔍 Querying Pinecone index '{index_name}'...")

        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        print(f" Pinecone returned {len(response.matches)} matches.")

        results = []
        for match in response.matches:
            score = match.score
            text = match.metadata.get("text", "")
            print(f"• Match Score: {score:.4f} | Text: {text[:100]}...")

            results.append(Document(page_content=text, metadata={"score": score}))

        return results

     except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        return []
        

   

'''class Milvus:
    
    def __init__(self):
        self.url = os.getenv("MILVUS_URL","http://localhost:19530")
        self.client = MilvusClient(self.url)
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
    @logger
    def create_database(self,db_name):
        try:
            self.client.create_database(db_name=db_name)
        except Exception as e:
            raise e
    
    @logger
    def list_db(self):
        return self.client.list_databases()
    
    @logger
    def discribe_db(self, db_name):
        try:
            return self.client.describe_database(db_name="default")
        except Exception as e:
            raise e
    
    @logger
    def create_schema(self,Schema_details : list[dict],index_type: str ,Drop_collection : bool = False):
        if self.collection_name not in self.client.list_collections() or Drop_collection is True:
            try:
                self.client.drop_collection(
                            collection_name=self.collection_name
                        )
                schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )
                index_filed_name = "embedding"
                
                for s in Schema_details:
                    schema.add_field(**s)
                    
                if index_filed_name:
                    index_data = milvus_indexes.get(index_type)
                    index_params = self.client.prepare_index_params()
                    index_params.add_index(
                        field_name=index_filed_name,
                        metric_type=index_data.get('metic_type'),
                        index_type=index_data.get('index_type'),
                        index_name=index_data.get("index_name"),
                        params=index_data.get("params")
                    )

                self.client.create_collection(
                        collection_name=self.collection_name,
                        schema=schema,
                        index_params=index_params
                    )
            except Exception as e:
                raise e
        
        else:
            print("the Collection already exists. Try a differnt name or drop_collection = True")
    
    @logger
    def load_collection(self):
        self.client.load_collection(collection_name=self.collection_name)
        pass

    @logger
    def insert_data(self,text_chunks : list, chunk_embeding : list):

        data = []

        for t,e in zip(text_chunks,chunk_embeding):
            info = {
                "pk": uuid.uuid4(),
                "embeedings" : e ,
                "text" : t
            }
        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
        except Exception as e:
            raise e'''


    # def store_data():
    #     pass
    # def retrive_data():
    #     pass

class PGvectordb:
    def store_data():
        pass
    def retrive_data():
        pass






   