from gensim.models import Word2Vec
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
    
    
    @staticmethod
    def word2vec():
        # Initilize and train the model
        sentences = [["This", "is", "an", "example", "Harman"], ["Harman", "lives", "in", "San", "Jose"], ["It", "is", "a", "sunny", "day"]]
        model = Word2Vec(sentences, min_count=1)
    
        # Access a word vector that is in the vocabulary
        try:
            vector_harman = model.wv['San']
            print("Vector for 'Harman':")
            print(vector_harman)
        except KeyError as e:
            print(e)
    
    # You can also check the vocabulary
    print("\nModel vocabulary:")
    print(list(model.wv.index_to_key))
    
    
if __name__ == "__main__":
    #embed_model.huggingface_embedding()
    embed_model.word2vec()
