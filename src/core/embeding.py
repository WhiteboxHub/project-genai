from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils.logger import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class sentence_transformer_embeding_model(Embeddings):
            def __init__(self, s_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" ):
                self.model_name = s_model
                self.model = SentenceTransformer(self.model_name)
            
            def embed_documents(self,text : list):
                embed_doc = []
                for doc in tqdm(text):
                    embed_doc.append(self.model.encode(doc).tolist())
                return embed_doc
            
            def embed_query(self,query : str):
                return self.model.encode([query])[0].tolist()
            
class Huggginface_model(Embeddings):
    def __init__(self, s_model ="BAAI/bge-small-en-v1.5"):
        self.model_name = s_model
        self.embed_model = HuggingFaceEmbedding(self.model_name)
    
    def embed_documents(self,text : list):
        embed_doc = []
        for doc in tqdm(text):
            embed_doc.append(self.embed_model.get_text_embedding(doc))
        return embed_doc
    
    def embed_query(self,query : str):
        return self.embed_model.get_text_embedding(query)
class embed_model:

    @logger
    @staticmethod
    def sentence_transformer(model_name):
        class sentence_transformer_embeding_model(Embeddings):
            def __init__(self, s_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" ):
                self.model_name = s_model
                self.model = SentenceTransformer(self.model_name)
            
            def embed_documents(self,text : list):
                embed_doc = []
                for doc in tqdm(text):
                    embed_doc.append(self.model.encode(doc).tolist())
                return embed_doc
            
            def embed_query(self,query : str):
                return self.model.encode([query])[0].tolist()

         
        return sentence_transformer_embeding_model(model_name)
    

    @logger
    @staticmethod
    def huggingface_embeding(model_name = "BAAI/bge-small-en-v1.5"):
        class Huggginface_model(Embeddings):
            def __init__(self, s_model ):
                self.model_name = s_model
                self.embed_model = HuggingFaceEmbedding(self.model_name)
            
            def embed_documents(self,text : list):
                embed_doc = []
                for doc in tqdm(text):
                    embed_doc.append(self.embed_model.get_text_embedding(doc))
                return embed_doc
            
            def embed_query(self,query : str):
                return self.embed_model.get_text_embedding(query)
        
        model = Huggginface_model(model_name)
        return model