from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
class embed_model:
    @staticmethod
    def sentence_Transfoer(chunks, model_name="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)
        return embeddings
    

    # def huggingface embeding
    pa