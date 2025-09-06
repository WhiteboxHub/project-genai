from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
class embed_model:    

    #SentenceTransformers
    @staticmethod
    def sentence_transformer_embed(text,model_name):
        model = SentenceTransformer(model_name)
        embeddings=model.encode(text)
        return embeddings
    
    #HuggingFace Embeddings
    @staticmethod
    def huggingface_embed(text,model):
        model = HuggingFaceEmbeddings(model_name=model)
        embeddings=model.embed_documents(text)
        return embeddings
        
        
    #print("SentenceTransformers embedding:\n", sentence_transformer_embed("hello world","all-MiniLM-L6-v2"))
    print("HuggingFace embedding:\n", huggingface_embed("its beautiful day","sentence-transformers/all-MiniLM-L6-v2"))
    
    
    