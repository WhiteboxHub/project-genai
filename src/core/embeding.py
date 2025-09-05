#from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import requests
import json
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()

class embed_model:
    @staticmethod
    def sentence_transformer():
        
        sentences = ["This is an example sentence", "Each sentence is converted"]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        print(embeddings.shape) # shape 384
        print(embeddings[0])
        #print(embeddings)

    @staticmethod
    def huggingface_embedding():
        client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN"],
        )

        result = client.feature_extraction(
            "Today is a sunny day and I will get some ice cream.",
            model="Qwen/Qwen3-Embedding-0.6B",
        )
        print(result)
    
    '''
    @staticmethod
    def word2vec():
        # Initilize and train the model
        model = Word2Vec(sentences,vector_size = 100, window = 5, min_count = 1)
        # Access a word vector
        vector_Alice = model.wv['Alice']
        #Save the mode
        model.save("my_word2vec_model.bin")
    
    '''
if __name__ == "__main__":
    embed_model.huggingface_embedding()
