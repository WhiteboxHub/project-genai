
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os


class embed_model:

    @staticmethod
    def sentence_Transformer():
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        # return embeddings
        print(embeddings)
    pass

    # def huggingface embeding
    @staticmethod
    def huggingface_embedding():
        api_key = os.getenv("HUGGINGFACE_API")
        if not api_key:
         raise RuntimeError("HUGGINGFACE_API is not set. Put it in your .env or environment.")
        client = InferenceClient(provider="hf-inference", api_key=api_key)
        result = client.feature_extraction(
         "Your text here",
        model="Qwen/Qwen3-Embedding-0.6B",)
        # return result
        print(result)
   
# Consider returning instead of printing: result


if __name__ == "__main__":
    # Example usage
    sentences = ["Hello world", "Embeddings are fun!"]
    embeddings = embed_model.sentence_transformer(sentences)
    print("SentenceTransformer Embeddings:", embeddings)

    # hf_result = embed_model.huggingface_embedding("This is a test text.")
    # print("HuggingFace Embedding:", hf_result)
    