from sentence_transformers import SentenceTransformer
import numpy as np

class embed_model:
    @staticmethod
    def sentence_Transformer(model_name='all-MiniLM-L6-v2'):
        """
        Load a sentence transformer model for embeddings
        """
        try:
            if not model_name.startswith('sentence-transformers/'):
                model_name = f'sentence-transformers/{model_name}'
            
            model = SentenceTransformer(model_name)
            print(f"✓ Loaded model: {model_name}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None

    @staticmethod
    def get_embeddings(model, texts):
        """
        Generate embeddings for a list of texts
        """
        if model is None:
            print("✗ Model not loaded")
            return None
        
        try:
            embeddings = model.encode(texts)
            print(f"✓ Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            return None

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

if __name__ == "__main__":

    texts = [
        "I love machine learning",
        "Artificial intelligence is amazing",
        "The weather is nice today"
    ]
    

    model = embed_model.sentence_Transformer()
    
    
    if model:
        embeddings = embed_model.get_embeddings(model, texts)
        
        if embeddings is not None:
            print(f"Embedding shape: {embeddings.shape}")
            print(f"First embedding sample: {embeddings[0][:5]}...")  
            
            similarity = embed_model.cosine_similarity(embeddings[0], embeddings[1])
            print(f"Similarity between first two sentences: {similarity:.3f}")