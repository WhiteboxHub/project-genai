
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
    
class embed_model:     
    
    @staticmethod
    def sentence_Transfer():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(sentence, convert_to_numpy=True)
        return embedding
    
    sent = "Artificial intelligence is transforming the world."
    embedding = sentence_transfer(sent)

    print(f"Sentence: {sent}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
        
    @staticmethod
    def huggingface_embeding(sentence: str):
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    
    # Forward pass
        with torch.no_grad():
           outputs = model(**inputs)
    
    # Mean pooling (average token embeddings)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()  # shape: (embedding_dim,)
    
sent = "Hugging Face embeddings are great for semantic search."
embedding = huggingface_embedding(sent)

print(f"Embedding shape: {embedding.shape}")
print(f"First 10 values: {embedding[:10]}")
    