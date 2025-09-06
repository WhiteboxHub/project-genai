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
        
        
    print("SentenceTransformers embedding:\n", sentence_transformer_embed("hello world","all-MiniLM-L6-v2"))
    #print("HuggingFace embedding:\n", huggingface_embed("its beautiful day","sentence-transformers/all-MiniLM-L6-v2"))
    
    
    #output
    '''
SentenceTransformers embedding:
 [-3.44772749e-02  3.10231782e-02  6.73497003e-03  2.61089858e-02
 -3.93620245e-02 -1.60302445e-01  6.69240132e-02 -6.44148979e-03
 -4.74504791e-02  1.47588560e-02  7.08752796e-02  5.55276312e-02
  1.91933345e-02 -2.62513123e-02 -1.01095429e-02 -2.69404557e-02
  2.23074611e-02 -2.22266484e-02 -1.49692640e-01 -1.74930077e-02......]
    '''