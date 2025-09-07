
from sentence_transformers import SentenceTransformer
from langchain
class embed_model:
 
    @staticmethod
    def sentence_Transform():
  
        sentences = ["This is an example sentence", "Each sentence is converted"]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        print(embeddings)

    @staticmethod
    def huggingface_embeddings():
        
    # def huggingface embeding
    pass