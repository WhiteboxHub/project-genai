from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
class embed_model:
    @staticmethod
    def sentence_Transfoer(chunks, model_name="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)
        return embeddings
    
    @staticmethod
    def huggingface_embedding(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
     embedding_model=HuggingFaceEmbeddings(model_name)   
     embedding = embedding_model.embed_documents(chunks)
     return embedding
    